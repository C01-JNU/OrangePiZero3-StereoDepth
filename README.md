本项目代码由deepseek编写
第三方库包括:
sudo apt install libspdlog-dev

注意:
1.config/global_config.yaml中的参数只会在编译时应用，若修改了其参数，则请重新编译
2.输入的双目图像应当是这样的格式：
若双目摄像头输入图像总分辨率AxB，则单眼分辨率是(A/2)xB且应当是左右拼接的
否则需要另设程序来转化
3.拍摄标定图像的时候最好要:
i)棋盘格水平
ii)多距离、多位置
iii)相机对准棋盘格中心
可以拍摄好后使用tools/split_stereo_images.sh来自动左右分割、命名图像
将标定用的图像放到指定位置(默认images/calibration)后，使用程序./build/bin/stereo_calibrator即可进行标定
