本项目代码由deepseek编写
第三方库包括:
sudo apt install libspdlog-dev

注意:
config/global_config.yaml中的参数只会在编译时应用，若修改了其参数，则请重新编译
输入的双目图像应当是这样的格式：
若双目摄像头输入图像分辨率为AxB，则双目图像应当由左右眼AxB压缩成(A/2)x(B/2)后左右拼接而成AxB的双目图像，否则需要自行转换
