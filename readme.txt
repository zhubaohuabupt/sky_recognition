识别天空的代码接口在Remove_sky.h里面

get_img_without_sky_opt(cv::Mat souceimg,cv::Mat &out)



使用说明 (在vs下开发的)：
1 依赖：opencv 
        Eigen 


2 输入 Mat类型：支持单通道和三通道
   输出Mat类型：单通道
  新建一个mainSky.cpp

#include"Remove_sky.h"
void main{
 Mat souceimg = imread("test.png",0);	
 Mat out;																					
  get_img_without_sky_opt( souceimg ,out);
  imshow(" 去除天空后的图像",out);
  cvWaitKey(0);

}
