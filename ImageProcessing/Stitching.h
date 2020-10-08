#include <opencv2/xfeatures2d/nonfree.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/stitching.hpp>

using namespace cv;
using namespace cv::xfeatures2d;
using namespace std;

typedef struct
{
	Point2f left_top;
	Point2f left_bottom;
	Point2f right_top;
	Point2f right_bottom;
}_Four_Corners;

static Mat homo;
static _Four_Corners corners;

void CalcCorners(const Mat& H, const Mat& src);
void OptimizeSeam(Mat& img1, Mat& trans, Mat& dst);
Mat imageSpilceBySURF(Mat& left, Mat& right);
Mat imageSpilceByStitching(Mat& left, Mat& right);
Mat imageSpilceBySURFforFixed(Mat& left, Mat& right);

