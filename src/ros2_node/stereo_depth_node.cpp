#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <sensor_msgs/point_cloud2_iterator.hpp>
#include <cv_bridge/cv_bridge.hpp>
#include <image_transport/image_transport.hpp>
#include <memory>
#include <chrono>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <filesystem>
#include <unistd.h>
#include <Eigen/Core>

#include "utils/logger.hpp"
#include "utils/config.hpp"
#include "calibration/stereo_rectifier.hpp"
#include "calibration/calibration_loader.hpp"

#if ENABLE_CPU
#include "cpu_stereo/cpu_stereo_matcher.hpp"
#endif

#if ENABLE_GPU
#include "vulkan/context.hpp"
#include "vulkan/stereo_pipeline.hpp"
#endif

using namespace stereo_depth::utils;
using namespace stereo_depth;

namespace stereo_depth_ros {

std::string getExeDir() {
    char result[PATH_MAX];
    ssize_t count = readlink("/proc/self/exe", result, PATH_MAX);
    if (count != -1) {
        result[count] = '\0';
        std::filesystem::path exePath(result);
        return exePath.parent_path().string();
    }
    return ".";
}

class StereoDepthNode : public rclcpp::Node {
public:
    StereoDepthNode() 
    : Node("stereo_depth_node"),
      target_fps_(15),        // 临时默认值，会被配置文件覆盖
      use_rectification_(false),
      publish_depth_(false),
      publish_pointcloud_(false),
      log_counter_(0)
    {
        Logger::initialize("stereo_depth_ros", spdlog::level::info);
        RCLCPP_INFO(this->get_logger(), "立体深度节点启动");

        std::string exeDir = getExeDir();
        std::string configPath;

        std::vector<std::string> candidatePaths = {
            (std::filesystem::path(exeDir) / "config" / "global_config.yaml").string(),
            (std::filesystem::path(exeDir).parent_path().parent_path() / "share" / "orangepizero3_stereodepth" / "config" / "global_config.yaml").string(),
            (std::filesystem::path(exeDir).parent_path().parent_path().parent_path() / "config" / "global_config.yaml").string()
        };

        bool configLoaded = false;
        for (const auto& path : candidatePaths) {
            if (std::filesystem::exists(path)) {
                configPath = path;
                RCLCPP_INFO(this->get_logger(), "找到配置文件: %s", configPath.c_str());
                configLoaded = true;
                break;
            }
        }

        if (!configLoaded) {
            RCLCPP_ERROR(this->get_logger(), "无法找到配置文件 global_config.yaml");
            rclcpp::shutdown();
            return;
        }

        auto& cfg_mgr = ConfigManager::getInstance();
        if (!cfg_mgr.loadGlobalConfig(configPath)) {
            RCLCPP_ERROR(this->get_logger(), "加载全局配置文件失败: %s", configPath.c_str());
            rclcpp::shutdown();
            return;
        }
        const auto& cfg = cfg_mgr.getConfig();

        // 从配置文件读取 ROS2 参数
        this->declare_parameter<bool>("publish_depth", false);
        this->declare_parameter<bool>("publish_pointcloud", false);
        publish_depth_ = this->get_parameter("publish_depth").as_bool();
        publish_pointcloud_ = this->get_parameter("publish_pointcloud").as_bool();

        RCLCPP_INFO(this->get_logger(), "ROS2 参数: publish_depth=%d, publish_pointcloud=%d",
                    publish_depth_, publish_pointcloud_);

        std::string left_topic = cfg.get<std::string>("ros2.topics.left_image", "/camera/left/image_raw");
        std::string right_topic = cfg.get<std::string>("ros2.topics.right_image", "/camera/right/image_raw");
        std::string disparity_topic = cfg.get<std::string>("ros2.topics.disparity", "/stereo/disparity");
        std::string depth_topic = cfg.get<std::string>("ros2.topics.depth", "/stereo/depth");
        std::string pointcloud_topic = cfg.get<std::string>("ros2.topics.pointcloud", "/stereo/points");
        int qos_depth = cfg.get<int>("ros2.qos_depth", 10);
        std::string qos_reliability = cfg.get<std::string>("ros2.qos_reliability", "best_effort");

        target_fps_ = cfg.get<int>("performance.target_fps", 15);
        use_rectification_ = cfg.get<bool>("calibration.rectify_images", false);
        std::string calib_file = cfg.get<std::string>("calibration.calibration_file", "calibration_results/stereo_calibration.yml");

        std::string calibPath;
        std::vector<std::string> calibCandidates = {
            (std::filesystem::path(exeDir) / "calibration_results" / "stereo_calibration.yml").string(),
            (std::filesystem::path(exeDir).parent_path().parent_path() / "share" / "orangepizero3_stereodepth" / "calibration_results" / "stereo_calibration.yml").string(),
            (std::filesystem::path(exeDir).parent_path().parent_path().parent_path() / "calibration_results" / "stereo_calibration.yml").string()
        };

        bool calibFound = false;
        for (const auto& path : calibCandidates) {
            if (std::filesystem::exists(path)) {
                calibPath = path;
                calibFound = true;
                break;
            }
        }

        if (use_rectification_) {
            if (!calibFound) {
                RCLCPP_ERROR(this->get_logger(), "无法找到标定文件，将跳过校正");
                use_rectification_ = false;
            } else {
                calibration::CalibrationParams params;
                calibration::CalibrationLoader loader;
                if (loader.loadFromFile(calibPath, params)) {
                    rectifier_ = std::make_unique<calibration::StereoRectifier>();
                    if (rectifier_->initialize(params, calibration::RectificationMode::SCALE_TO_FIT)) {
                        RCLCPP_INFO(this->get_logger(), "立体校正器初始化成功");
                    } else {
                        RCLCPP_ERROR(this->get_logger(), "立体校正器初始化失败，将跳过校正");
                        rectifier_.reset();
                    }
                } else {
                    RCLCPP_ERROR(this->get_logger(), "加载标定文件失败，将跳过校正");
                }
            }
        }

#if ENABLE_CPU
        cpu_matcher_ = std::make_unique<cpu_stereo::CpuStereoMatcher>();
        if (!cpu_matcher_->initializeFromConfig()) {
            RCLCPP_ERROR(this->get_logger(), "CPU 匹配器初始化失败");
            rclcpp::shutdown();
            return;
        }
        RCLCPP_INFO(this->get_logger(), "CPU 立体匹配引擎已初始化");
#elif ENABLE_GPU
        RCLCPP_ERROR(this->get_logger(), "GPU 模式暂未实现");
        rclcpp::shutdown();
        return;
#else
        RCLCPP_ERROR(this->get_logger(), "未启用任何深度计算模块");
        rclcpp::shutdown();
        return;
#endif

        auto qos = rclcpp::QoS(rclcpp::KeepLast(qos_depth));
        if (qos_reliability == "reliable") {
            qos.reliable();
        } else {
            qos.best_effort();
        }

        auto callback_group = this->create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);
        rclcpp::SubscriptionOptions options;
        options.callback_group = callback_group;

        left_sub_ = this->create_subscription<sensor_msgs::msg::Image>(
            left_topic, qos,
            std::bind(&StereoDepthNode::leftImageCallback, this, std::placeholders::_1),
            options);
        right_sub_ = this->create_subscription<sensor_msgs::msg::Image>(
            right_topic, qos,
            std::bind(&StereoDepthNode::rightImageCallback, this, std::placeholders::_1),
            options);

        disp_pub_ = image_transport::create_publisher(this, disparity_topic);

        if (publish_depth_) {
            depth_pub_ = this->create_publisher<sensor_msgs::msg::Image>(depth_topic, qos);
        }
        if (publish_pointcloud_) {
            cloud_pub_ = this->create_publisher<sensor_msgs::msg::PointCloud2>(pointcloud_topic, qos);
        }

        auto period = std::chrono::milliseconds(1000 / target_fps_);
        timer_ = this->create_wall_timer(period, std::bind(&StereoDepthNode::processFrame, this));

        RCLCPP_INFO(this->get_logger(), "节点初始化完成，目标帧率: %d FPS", target_fps_);
    }

private:
    void leftImageCallback(const sensor_msgs::msg::Image::SharedPtr msg) {
        std::lock_guard<std::mutex> lock(left_mutex_);
        left_image_ = msg;
    }

    void rightImageCallback(const sensor_msgs::msg::Image::SharedPtr msg) {
        std::lock_guard<std::mutex> lock(right_mutex_);
        right_image_ = msg;
    }

    void processFrame() {
        sensor_msgs::msg::Image::SharedPtr left_msg, right_msg;
        {
            std::lock_guard<std::mutex> left_lock(left_mutex_);
            std::lock_guard<std::mutex> right_lock(right_mutex_);
            left_msg = left_image_;
            right_msg = right_image_;
        }

        if (!left_msg || !right_msg) {
            RCLCPP_DEBUG(this->get_logger(), "等待图像...");
            return;
        }

        cv::Mat left_img, right_img;
        try {
            left_img = cv_bridge::toCvCopy(left_msg, "mono8")->image;
            right_img = cv_bridge::toCvCopy(right_msg, "mono8")->image;
        } catch (const std::exception& e) {
            RCLCPP_ERROR(this->get_logger(), "图像转换失败: %s", e.what());
            return;
        }

        cv::Mat left_rect, right_rect;
        if (rectifier_) {
            if (!rectifier_->rectifyPair(left_img, right_img, left_rect, right_rect)) {
                RCLCPP_WARN(this->get_logger(), "校正失败，跳过此帧");
                return;
            }
        } else {
            left_rect = left_img;
            right_rect = right_img;
        }

        cv::Mat disparity;
        auto start = std::chrono::high_resolution_clock::now();
#if ENABLE_CPU
        disparity = cpu_matcher_->compute(left_rect, right_rect);
#endif
        auto end = std::chrono::high_resolution_clock::now();
        float proc_ms = std::chrono::duration<float, std::milli>(end - start).count();
        RCLCPP_DEBUG(this->get_logger(), "视差计算耗时: %.2f ms", proc_ms);

        publishDisparity(disparity, left_msg->header.stamp);

        if (publish_depth_ && rectifier_) {
            cv::Mat depth = computeDepth(disparity);
            publishDepth(depth, left_msg->header.stamp);
        }

        if (publish_pointcloud_ && rectifier_) {
            auto cloud = generatePointCloudManual(disparity, left_msg->header.stamp);
            cloud_pub_->publish(cloud);
        }
    }

    void publishDisparity(const cv::Mat& disparity, const rclcpp::Time& stamp) {
        cv::Mat disp_8u;
        double min_val, max_val;
        cv::minMaxLoc(disparity, &min_val, &max_val);
        disparity.convertTo(disp_8u, CV_8U, 255.0 / (max_val > 0 ? max_val : 64.0));

        auto disp_msg = cv_bridge::CvImage(std_msgs::msg::Header(), "mono8", disp_8u).toImageMsg();
        disp_msg->header.stamp = stamp;
        disp_msg->header.frame_id = "stereo_depth";
        disp_pub_.publish(disp_msg);
    }

    cv::Mat computeDepth(const cv::Mat& disparity) {
        const auto& params = rectifier_->getCalibrationParams();
        double fx = params.camera_matrix_left.at<double>(0, 0);
        double baseline = cv::norm(params.translation_vector); // 米

        cv::Mat depth(disparity.size(), CV_32FC1);
        for (int i = 0; i < disparity.rows; ++i) {
            for (int j = 0; j < disparity.cols; ++j) {
                float d = disparity.at<uint16_t>(i, j);
                if (d > 0) {
                    depth.at<float>(i, j) = static_cast<float>(fx * baseline / d);
                } else {
                    depth.at<float>(i, j) = 0.0f;
                }
            }
        }
        return depth;
    }

    void publishDepth(const cv::Mat& depth, const rclcpp::Time& stamp) {
        cv::Mat depth_16u;
        double min_val, max_val;
        cv::minMaxLoc(depth, &min_val, &max_val);
        depth.convertTo(depth_16u, CV_16U, 65535.0 / (max_val > 0 ? max_val : 1.0));

        auto depth_msg = cv_bridge::CvImage(std_msgs::msg::Header(), "mono16", depth_16u).toImageMsg();
        depth_msg->header.stamp = stamp;
        depth_msg->header.frame_id = "stereo_depth";
        depth_pub_->publish(*depth_msg);
    }

    sensor_msgs::msg::PointCloud2 generatePointCloudManual(const cv::Mat& disparity, const rclcpp::Time& stamp) {
        const auto& params = rectifier_->getCalibrationParams();
        double fx = params.camera_matrix_left.at<double>(0, 0);
        double fy = params.camera_matrix_left.at<double>(1, 1);
        double cx = params.camera_matrix_left.at<double>(0, 2);
        double cy = params.camera_matrix_left.at<double>(1, 2);
        double baseline = cv::norm(params.translation_vector); // 米

        // 将16位视差图转换为浮点型，并除以16得到真实视差（像素）
        cv::Mat disp_float;
        disparity.convertTo(disp_float, CV_32F, 1.0/16.0);

        // 计算视差非零像素总数，用于调试
        int total_pixels = disp_float.rows * disp_float.cols;
        int valid_pixels = 0;

        // 临时存储点云数据
        std::vector<Eigen::Vector3f> points;
        points.reserve(total_pixels);

        // 记录第一个有效点的信息
        bool first_valid_logged = false;

        for (int v = 0; v < disp_float.rows; ++v) {
            for (int u = 0; u < disp_float.cols; ++u) {
                float d = disp_float.at<float>(v, u);
                if (d <= 0.5f) continue; // 忽略太小的视差（可能为0或噪声）

                valid_pixels++;

                // 深度计算
                float Z = fx * baseline / d;
                if (Z <= 0.001f) continue; // 忽略深度太小（可能为0）

                float X = (u - cx) * Z / fx;
                float Y = (v - cy) * Z / fy;

                points.emplace_back(X, Y, Z);

                // 打印第一个有效点（仅在前5帧内）
                if (log_counter_ < 5 && !first_valid_logged) {
                    RCLCPP_INFO(this->get_logger(), "第一个有效点: u=%d, v=%d, d=%f, X=%f, Y=%f, Z=%f",
                                u, v, d, X, Y, Z);
                    first_valid_logged = true;
                }
            }
        }

        // 打印统计信息（仅前5帧）
        if (log_counter_ < 5) {
            RCLCPP_INFO(this->get_logger(), "点云统计: 总像素=%d, 有效像素(视差>0.5)=%d, 生成点数=%zu",
                        total_pixels, valid_pixels, points.size());
            log_counter_++;
        }

        // 转换为ROS消息
        sensor_msgs::msg::PointCloud2 cloud_msg;
        sensor_msgs::PointCloud2Modifier modifier(cloud_msg);
        modifier.setPointCloud2FieldsByString(1, "xyz");
        modifier.resize(points.size());  // 直接设置大小，并更新 width/height

        sensor_msgs::PointCloud2Iterator<float> iter_x(cloud_msg, "x");
        sensor_msgs::PointCloud2Iterator<float> iter_y(cloud_msg, "y");
        sensor_msgs::PointCloud2Iterator<float> iter_z(cloud_msg, "z");

        for (const auto& p : points) {
            *iter_x = p.x();
            *iter_y = p.y();
            *iter_z = p.z();
            ++iter_x; ++iter_y; ++iter_z;
        }

        cloud_msg.header.stamp = stamp;
        cloud_msg.header.frame_id = "stereo_depth";

        // 调试输出消息元数据
        RCLCPP_DEBUG(this->get_logger(), "点云消息: width=%d, height=%d, point_step=%d, row_step=%d",
                     cloud_msg.width, cloud_msg.height,
                     cloud_msg.point_step, cloud_msg.row_step);

        return cloud_msg;
    }

    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr left_sub_, right_sub_;
    image_transport::Publisher disp_pub_;
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr depth_pub_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr cloud_pub_;
    rclcpp::TimerBase::SharedPtr timer_;

    sensor_msgs::msg::Image::SharedPtr left_image_, right_image_;
    std::mutex left_mutex_, right_mutex_;

    std::unique_ptr<calibration::StereoRectifier> rectifier_;
#if ENABLE_CPU
    std::unique_ptr<cpu_stereo::CpuStereoMatcher> cpu_matcher_;
#endif

    int target_fps_;
    bool use_rectification_;
    bool publish_depth_;
    bool publish_pointcloud_;
    int log_counter_; // 用于控制前几帧的日志输出
};

} // namespace stereo_depth_ros

int main(int argc, char* argv[]) {
    rclcpp::init(argc, argv);
    auto node = std::make_shared<stereo_depth_ros::StereoDepthNode>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}
