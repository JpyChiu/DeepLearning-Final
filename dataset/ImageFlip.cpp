#include <opencv2/opencv.hpp>
#include <iostream>
#include <math.h>
#include <sstream>

using namespace cv;
using namespace std;



Mat matRotateClockWise90(Mat src)
{
	// 矩阵转置
	transpose(src, src);
	//0: 沿X轴翻转； >0: 沿Y轴翻转； <0: 沿X轴和Y轴翻转
	flip(src, src, 1);// 翻转模式，flipCode == 0垂直翻转（沿X轴翻转），flipCode>0水平翻转（沿Y轴翻转），flipCode<0水平垂直翻转（先沿X轴翻转，再沿Y轴翻转，等价于旋转180°）
	return src;
}

Mat matRotateClockWise180(Mat src)//顺时针180
{
	//0: 沿X轴翻转； >0: 沿Y轴翻转； <0: 沿X轴和Y轴翻转
	flip(src, src, 0);// 翻转模式，flipCode == 0垂直翻转（沿X轴翻转），flipCode>0水平翻转（沿Y轴翻转），flipCode<0水平垂直翻转（先沿X轴翻转，再沿Y轴翻转，等价于旋转180°）
	flip(src, src, 1);
	return src;
	//transpose(src, src);// 矩阵转置
}

Mat matRotateClockWise270(Mat src)//顺时针270
{
	// 矩阵转置
	//transpose(src, src);
	//0: 沿X轴翻转； >0: 沿Y轴翻转； <0: 沿X轴和Y轴翻转
	transpose(src, src);// 翻转模式，flipCode == 0垂直翻转（沿X轴翻转），flipCode>0水平翻转（沿Y轴翻转），flipCode<0水平垂直翻转（先沿X轴翻转，再沿Y轴翻转，等价于旋转180°）
	flip(src, src, 0);
	return src;
}

int main()
{
	Mat  src_img, gray_img, binary_img;
	string s,s1;
	stringstream ss;
	for (int i = 1; i <=500; i++)
	{
		if (i < 10)
			s = "00000";
		else if (i < 100)
			s = "0000";
		else
			s = "000";
		s1 = "";
		ss << i;
		ss >> s1;
		ss.str("");
		ss.clear();
		src_img = imread(s + s1 + ".jpg");
		resize(src_img, src_img, Size(224, 224), (0, 0), (0, 0), INTER_LINEAR);
		imwrite(s + s1 + ".jpg", src_img);
		imwrite(s + s1 + "_1.jpg", matRotateClockWise90(src_img));
		imwrite(s + s1 + "_2.jpg", matRotateClockWise180(src_img));
		imwrite(s + s1 + "_3.jpg", matRotateClockWise270(src_img));
	}
}