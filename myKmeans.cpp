/*
This file is part of remove_sky.
copyright：Zhu Baohua
2016.6
*/
#include"myKmeans.h"
#define k 2
using namespace std;  
 
//计算两个元组间的欧几里距离  
float getDistance(Vector3d t1, Vector3d t2)   
{  
    return sqrt((t1(0)- t2(0)) * (t1(0)- t2(0)) + (t1(1) - t2(1))* (t1(1) - t2(1))+(t1(2) - t2(2))*(t1(2) - t2(2)));  
}  
  
//根据质心，决定当前元组属于哪个簇  
int class_inputdata(Vector3d means[],Vector3d tuple){  
    float dist=getDistance(means[0],tuple);  
    float tmp;  
    int label=0;//标示属于哪一个簇  
    for(int i=1;i<k;i++){  
        tmp=getDistance(means[i],tuple);  
        if(tmp<dist) {dist=tmp;label=i;}  
    }  
    return label;     
}  
//获得给定簇集的平方误差  
float getVar(vector<Vector3d> clusters[],Vector3d means[])
{  
    float var = 0;
	float count=0;
    for (int i = 0; i < k; i++)  
    {  
        vector<Vector3d> t = clusters[i];  
        for (int j = 0; j< t.size(); j++)  
        {  count++;
            var += getDistance(t[j],means[i]);  
        }  
    }  
    //cout<<"sum:"<<sum<<endl;  
    return var/count;  
  
}  
//获得当前簇的均值（质心）  
Vector3d getMeans(vector<Vector3d> cluster){  
      
    double num = cluster.size();  
    Vector3d t;  
    for (double i = 0; i < num; i++)  
    {																						//	cout<<"cluster: "<<cluster[i]<<endl;
        t(0) += cluster[i](0);  
        t(1) += cluster[i](1);  
		t(2) += cluster[i](2);  
    }  
	    t(0)/=num;
		  t(1)/=num;
		    t(2)/=num;
    return t;  

}  
bool check_repeat(Vector3d means[],int tmp)
{
	for(int i=tmp;i>=1;i--)
	{
	if(means[tmp]==means[i-1])
		return true;
	}
	return false;
}
void myKMeans(Mat inputdata,Mat &outlabel, Vector3d centers[])
{  
    vector<Vector3d> clusters[k];  
 																									
    //默认一开始将前K个元组的值作为k个簇的质心（均值）  
    for(int i=0;i<k;i++)
	  {  
        centers[i](0)=inputdata.at<Vec3b>(i,0)(0);  
        centers[i](1)=inputdata.at<Vec3b>(i,0)(1);   
		centers[i](2)=inputdata.at<Vec3b>(i,0)(2); 
		if(i>0)
		{        int t=1;
				while(check_repeat( centers, i))
				{
				 centers[i](0)=inputdata.at<Vec3b>(i+t,0)(0);  
				centers[i](1)=inputdata.at<Vec3b>(i+t,0)(1);   
				centers[i](2)=inputdata.at<Vec3b>(i+t,0)(2); 
				t++;
				}
		}
		//cout<< "centers[i]  "<< centers[i]<<endl;
    }  
    int lable=0;  
    //根据默认的质心给簇赋值  
	int width=inputdata.cols;
	int height=inputdata.rows;
	double data_num=height;//列数据
																							//cout<<"data_num : "<<data_num<<endl;
    for( int i=0;i!=data_num;++i){  
		Vector3d tmp;
	    tmp(0)=inputdata.at<Vec3b>(i,0)(0);  
        tmp(1)=inputdata.at<Vec3b>(i,0)(1);   
		tmp(2)=inputdata.at<Vec3b>(i,0)(2);   
        lable=class_inputdata(centers,tmp);  
																				/*if(lable==1)
																				cout<<" lable : "<<lable<<endl;*/
        clusters[lable].push_back(tmp);  
    }  
																				//输出刚开始的簇  
																				/*for(lable=0;lable<k;lable++){  
																					cout<<"第"<<lable+1<<"个簇："<<endl;  
																					cout<<clusters[lable].size()<<endl;
																					 }  */
    float oldVar=-1;  
    float newVar=getVar(clusters,centers);
																							//	cout<<"newVar  "<<newVar<<endl;
	int count_=0;
    while(abs(newVar - oldVar) >= 1) //当新旧函数值相差不到1即准则函数值不发生明显变化时，算法终止  
      {  
																						// cout<<" count_ "<<count_++<<endl;
        for (int i = 0; i < k; i++) //更新每个簇的中心点  
        {  
            centers[i] = getMeans(clusters[i]);  //////////////
																							//			 cout<<"centers "<<centers[i]<<endl;  
        }  
        oldVar = newVar;  
        newVar = getVar(clusters,centers); //计算新的准则函数值  
																							//	cout<<"newVar  "<<newVar<<endl;

        for (int i = 0; i < k; i++) //清空每个簇  
          {  
             clusters[i].clear();  
          }  
        //根据新的质心获得新的簇  
        for(int i=0;i!=data_num;++i)
		 {  			
						Vector3d tmp;
					tmp(0)=inputdata.at<Vec3b>(i,0)(0);  
					tmp(1)=inputdata.at<Vec3b>(i,0)(1);   
					tmp(2)=inputdata.at<Vec3b>(i,0)(2);   
					lable=class_inputdata(centers,tmp);  
					clusters[lable].push_back(tmp); 
					
					outlabel.at<uchar>(i,0)=lable;
        }  
           
      }																		 /*
																			for(lable=0;lable<k;lable++){  
																					cout<<"第"<<lable+1<<"个簇："<<endl;  
																					cout<<clusters[lable].size()<<endl; 
		
																			}  */
}     
 
