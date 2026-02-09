本项目代码由deepseek编写
第三方库包括:
sudo apt install libspdlog-dev

注意:
config/global_config.yaml中的参数只会在编译时应用，若修改了其参数，则请重新编译
输入的双目图像应当是这样的格式：
若双目摄像头输入图像总分辨率AxB，则单眼分辨率是(A/2)xB且应当是左右拼接的
否则需要另设程序来转化
