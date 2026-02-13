本项目代码由deepseek编写
第三方库包括:
sudo apt install libspdlog-dev

前序:
目前实测cpu模式的sgbm算法能以50ms/张左右的速度处理好640x480的双目拼接图像,占用约70%
但是gpu需要1700ms左右
只能说香橙派zero3的GPU性能真的太差了，不好用
现在基本上除了cpu_sgbm之外都是玩具，没用，所以现在配置文件就配置cpu模式的sgbm了

注意:
1.config/global_config.yaml中的参数只会在编译时应用，若修改了其参数，则请重新编译
若是变更了GPU的相关设置，则需要把src/vulkan里面的spv文件夹和generated文件夹手动删除
2.输入的双目图像应当是这样的格式：
若双目摄像头输入图像总分辨率AxB，则单眼分辨率是(A/2)xB且应当是左右拼接的
否则需要另设程序来转化
3.拍摄标定图像的时候最好要:
i)棋盘格水平
ii)多距离、多位置
iii)相机对准棋盘格中心
可以拍摄好后使用tools/split_stereo_images.sh来自动左右分割、命名图像
将标定用的图像放到指定位置(默认images/calibration)后，使用程序./build/bin/stereo_calibrator即可进行标定
4.目前程序都需要在项目根目录运行
