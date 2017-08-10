# sky_recognition
This project is a demo of sky recognition.

How to use ?

1 dependency：opencv and Eigen  

I develop this project in Visual studio,so you will be easily test it in VS. also you build a cmake projet in linux.

 
2 launch the project 
you can find the function  'get_img_without_sky_opt(cv::Mat souceimg,cv::Mat &out)' in Remove_sky.h.
just  use it as follows:
#include"Remove_sky.h"
void main{
 Mat souceimg = imread("test.png",0);	
 Mat out;																					
  get_img_without_sky_opt( souceimg ,out);
  imshow(" out ",out);
  cvWaitKey(0);
}
