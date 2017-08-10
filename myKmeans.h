/*
This file is part of remove_sky.
copyright£ºZhu Baohua
2016.6
*/
#include <iostream>  
#include <fstream>  
#include <vector>  
#include <math.h>  
#include<highgui.h>
#include<opencv.hpp>
#include <Eigen/Dense>    
#include <Eigen/LU>  
using namespace cv;  
using namespace Eigen;
void myKMeans(Mat inputdata,Mat &outlabel, Vector3d centers[]);