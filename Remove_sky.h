/*
This file is part of remove_sky.
copyright：Zhu Baohua
2016.6
*/

#include <iostream>    
#include <Eigen/Dense>   
#include <Eigen/LU>  
#include<vector>
#include<deque>
#include <cmath>
#include <highgui.h>
#include<opencv2/opencv.hpp>
#include "opencv2/core/core.hpp" 
using Eigen::MatrixXd;  
using namespace std;
using namespace cv;
using namespace Eigen;

float get_max_3(float a,float b,float c);
//
int border_avg(vector<int> outborder);
int absolute_border_dif(vector<int> outborder);
//给定一个梯度阈值，遍历图像求出一边界
void Calculate_border(cv::Mat grad,vector<int>&outborder,int grad_threshold);
//把天空区域涂黑
void get_no_sky_img( cv::Mat& SourceImg,vector<int> outborder);
//计算天空和地面的像素灰度的平均值
void get_vector3d_mean(cv::Mat SourceImg,vector<int>border_tmp,Eigen::Vector3d &sky_meanRGB,Eigen::Vector3d &ground_meanRGB);
//给定一个 border ，计算天空和地面像素点数
void get_sky_num(cv::Mat SourceImg,vector<int>border_tmp,long int &num_sky,long int &num_ground);
//计算搜索采样点数
int get_sample_pointnum(int threshold_min,int  threshold_max,int search_step);
//计算变化梯度阈值t
int get_grad_threahold_t(int threshold_min,int  threshold_max,int sample_point_num,int k);
//每次给定一个border，计算其能量函数
 double get_Jnt(cv::Mat SourceImg, Eigen::Vector3d &sky_meanRGB,Eigen::Vector3d &ground_meanRGB,vector<int>border_tmp);
 //得到去除天空的图片  
 // input：梯度搜索最小值、最大值、搜索间隔、输入的灰度图像
 // output: 去除天空的图像

int get_img_without_sky_opt(cv::Mat grayImg,cv::Mat &no_sky_img);
void get_img_without_sky(int threshold_min,int  threshold_max,int search_step,cv::Mat souceimg,cv::Mat &out);
//计算马氏距离
double cal_vector3d_Mahalanobis(Eigen::Vector3d input1 ,Eigen::Vector3d input2);
//计算两个3维列向量的欧氏距离
double  Euclidean_distance(Eigen::Vector3d input1,Eigen::Vector3d input2);
//求优化后天空区域的像素均值
Vector3d cal_Vector3d_mean(cv::Mat channel3img,cv::Mat out);
//把天空局域分为两类，并在灰度图片上显示出来
void classify_sky_byK_means(vector<int> bopt ,cv::Mat& class_sky,cv::Mat channel3img);
//论文中算法三，重新计算天空边界
void Sky_borde_recalculation(vector<int> bopt ,cv::Mat class_sky,cv::Mat channel3img,cv::Mat&out);
void low_textrue_check(cv::Mat inputimg,cv::Mat&outputimg,int SizeN);