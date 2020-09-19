#include <opencv2/xfeatures2d/nonfree.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/stitching.hpp>
#include <time.h>
#include <iostream>  

using namespace cv;
using namespace cv::xfeatures2d;
using namespace std;

void OptimizeSeam(Mat& img1, Mat& trans, Mat& dst);
Mat imageSpilceBySURF(Mat& left, Mat& right);
Mat imageSpilceByStitching(Mat& left, Mat& right);
Mat imageSpilceBySURFforFixed(Mat& left, Mat& right);
Mat homo;

typedef struct
{
	Point2f left_top;
	Point2f left_bottom;
	Point2f right_top;
	Point2f right_bottom;
}four_corners_t;

four_corners_t corners;

void CalcCorners(const Mat& H, const Mat& src)
{
	double v2[] = { 0, 0, 1 };
	double v1[3];
	Mat V2 = Mat(3, 1, CV_64FC1, v2);
	Mat V1 = Mat(3, 1, CV_64FC1, v1);

	V1 = H * V2;
	corners.left_top.x = v1[0] / v1[2];
	corners.left_top.y = v1[1] / v1[2];

	v2[0] = 0;
	v2[1] = src.rows;
	v2[2] = 1;
	V2 = Mat(3, 1, CV_64FC1, v2);
	V1 = Mat(3, 1, CV_64FC1, v1);
	V1 = H * V2;
	corners.left_bottom.x = v1[0] / v1[2];
	corners.left_bottom.y = v1[1] / v1[2];

	v2[0] = src.cols;
	v2[1] = 0;
	v2[2] = 1;
	V2 = Mat(3, 1, CV_64FC1, v2);
	V1 = Mat(3, 1, CV_64FC1, v1);
	V1 = H * V2;
	corners.right_top.x = v1[0] / v1[2];
	corners.right_top.y = v1[1] / v1[2];

	v2[0] = src.cols;
	v2[1] = src.rows;
	v2[2] = 1;
	V2 = Mat(3, 1, CV_64FC1, v2);
	V1 = Mat(3, 1, CV_64FC1, v1);
	V1 = H * V2;
	corners.right_bottom.x = v1[0] / v1[2];
	corners.right_bottom.y = v1[1] / v1[2];

}

int main(int argc, char* argv[])
{
	int64 e1, e2;
	double t;
	int count = 0;
	Mat img_left, img_right;
	VideoWriter writer;
	Mat dst;
	string filename = "result.avi";
	int img_height, img_width;

	clock_t start, end;
	double result;
	double fps;
	VideoCapture left_video("left.mp4");
	VideoCapture right_video("right.mp4");
	if ((!left_video.isOpened()) || (!right_video.isOpened()))
	{
		cout << "open failed" << endl;
		return -1;
	}

	start = clock();
	while ((left_video.read(img_left)) && (right_video.read(img_right)))
	{

		//dst = imageSpilceBySURF(img_left, img_right);
		dst = imageSpilceBySURFforFixed(img_left, img_right);
		//dst = imageSpilceByStitching(img_left, img_right);

		if (dst.empty())
			continue;
		if (count == 0) {
			int codec = VideoWriter::fourcc('X', 'V', 'I', 'D');
			fps = left_video.get(CAP_PROP_FPS);
			bool isColor = (img_left.type() == CV_8UC3);
			if (writer.open(filename, codec, fps, dst.size(), isColor)) {
				cout << "video success initialized" << endl;
			}
			img_height = dst.size().height;
			img_width = dst.size().width;
		}
		resize(dst, dst, Size(img_width, img_height));
		writer.write(dst);
		count += 1;

		cout << count << endl;
	}
	end = clock();
	result = (double)(end - start);
	cout << "FPS : " << fps << endl;
	cout << "정합 프레임 : " << count << endl;
	cout << "영상시간 : " << count / fps * 1000 << "ms" << endl;
	cout << "정합 소요시간 : " << result << "ms" << endl;
	return 0;
}


void OptimizeSeam(Mat& img1, Mat& trans, Mat& dst)
{
	int start = MIN(corners.left_top.x, corners.left_bottom.x);
	double processWidth = img1.cols - start;
	int rows = dst.rows;
	int cols = img1.cols;
	double alpha = 1;
	for (int i = 0; i < rows; i++)
	{
		uchar* p = img1.ptr<uchar>(i);
		uchar* t = trans.ptr<uchar>(i);
		uchar* d = dst.ptr<uchar>(i);
		for (int j = start; j < cols; j++)
		{
			if (t[j * 3] == 0 && t[j * 3 + 1] == 0 && t[j * 3 + 2] == 0)
			{
				alpha = 1;
			}
			else
			{
				alpha = (processWidth - (j - start)) / processWidth;
			}
			d[j * 3] = p[j * 3] * alpha + t[j * 3] * (1 - alpha);
			d[j * 3 + 1] = p[j * 3 + 1] * alpha + t[j * 3 + 1] * (1 - alpha);
			d[j * 3 + 2] = p[j * 3 + 2] * alpha + t[j * 3 + 2] * (1 - alpha);
		}
	}
}

Mat imageSpilceByStitching(Mat& left, Mat& right)
{
	Mat dst;
	Stitcher::Mode mode = Stitcher::PANORAMA;
	vector<Mat> imgs;
	if (left.empty() || right.empty()) {
		cout << "pic empty" << endl;
	}
	imgs.push_back(left);
	imgs.push_back(right);
	Ptr<Stitcher> stitcher = Stitcher::create(mode);
	Stitcher::Status status = stitcher->stitch(imgs, dst);
	if (status != Stitcher::OK)
	{
		cout << "Can't stitch images, error code = " << int(status) << endl;
	}
	return dst;

}

Mat imageSpilceBySURF(Mat& left, Mat& right)
{
	Mat img_gray_right, img_gray_left;
	cvtColor(right, img_gray_right, COLOR_RGB2GRAY);
	cvtColor(left, img_gray_left, COLOR_RGB2GRAY);

	/*특징점 추출*/
	Ptr<SURF> surf = SURF::create();
	vector<KeyPoint> key_point_right, key_point_left;
	surf->detect(img_gray_right, key_point_right);
	surf->detect(img_gray_left, key_point_left);
	Ptr<SurfDescriptorExtractor> Descriptor = SurfDescriptorExtractor::create();
	Mat imageDesc1, imageDesc2;
	Descriptor->compute(img_gray_right, key_point_right, imageDesc1);
	Descriptor->compute(img_gray_left, key_point_left, imageDesc2);

	/*매칭*/
	FlannBasedMatcher matcher;
	vector<vector<DMatch> > matchePoints;
	vector<DMatch> GoodMatchePoints;
	vector<Mat> train_desc(1, imageDesc1);
	matcher.add(train_desc);
	matcher.train();
	matcher.knnMatch(imageDesc2, matchePoints, 2);

	for (int i = 0; i < matchePoints.size(); i++)
	{
		if (matchePoints[i][0].distance < 0.4 * matchePoints[i][1].distance)
		{
			GoodMatchePoints.push_back(matchePoints[i][0]);
		}
	}


	vector<Point2f> imagePoints1, imagePoints2;
	for (int i = 0; i < GoodMatchePoints.size(); i++)
	{
		imagePoints2.push_back(key_point_left[GoodMatchePoints[i].queryIdx].pt);
		imagePoints1.push_back(key_point_right[GoodMatchePoints[i].trainIdx].pt);
	}

	Mat homo = findHomography(imagePoints1, imagePoints2, RANSAC);

	CalcCorners(homo, right);


	Mat imageTransform1, imageTransform2;
	warpPerspective(right, imageTransform1, homo, Size(MAX(corners.right_top.x, corners.right_bottom.x), left.rows));


	int dst_width = imageTransform1.cols;
	int dst_height = left.rows;
	Mat dst(dst_height, dst_width, CV_8UC3);
	dst.setTo(0);
	imageTransform1.copyTo(dst(Rect(0, 0, imageTransform1.cols, imageTransform1.rows)));

	left.copyTo(dst(Rect(0, 0, left.cols, left.rows)));
	OptimizeSeam(left, imageTransform1, dst);
	return dst;
}

Mat imageSpilceBySURFforFixed(Mat& left, Mat& right)
{
	if (homo.empty() == true) {
		Mat img_gray_right, img_gray_left;
		cvtColor(right, img_gray_right, COLOR_RGB2GRAY);
		cvtColor(left, img_gray_left, COLOR_RGB2GRAY);

		/*특징점 추출*/
		Ptr<SURF> surf = SURF::create();
		vector<KeyPoint> key_point_right, key_point_left;
		surf->detect(img_gray_right, key_point_right);
		surf->detect(img_gray_left, key_point_left);
		Ptr<SurfDescriptorExtractor> Descriptor = SurfDescriptorExtractor::create();
		Mat imageDesc1, imageDesc2;
		Descriptor->compute(img_gray_right, key_point_right, imageDesc1);
		Descriptor->compute(img_gray_left, key_point_left, imageDesc2);

		/*매칭*/
		FlannBasedMatcher matcher;
		vector<vector<DMatch> > matchePoints;
		vector<DMatch> GoodMatchePoints;
		vector<Mat> train_desc(1, imageDesc1);
		matcher.add(train_desc);
		matcher.train();
		matcher.knnMatch(imageDesc2, matchePoints, 2);

		for (int i = 0; i < matchePoints.size(); i++)
		{
			if (matchePoints[i][0].distance < 0.4 * matchePoints[i][1].distance)
			{
				GoodMatchePoints.push_back(matchePoints[i][0]);
			}
		}


		vector<Point2f> imagePoints1, imagePoints2;
		for (int i = 0; i < GoodMatchePoints.size(); i++)
		{
			imagePoints2.push_back(key_point_left[GoodMatchePoints[i].queryIdx].pt);
			imagePoints1.push_back(key_point_right[GoodMatchePoints[i].trainIdx].pt);
		}


		homo = findHomography(imagePoints1, imagePoints2, RANSAC);
	}

	CalcCorners(homo, right);


	Mat imageTransform1, imageTransform2;
	warpPerspective(right, imageTransform1, homo, Size(MAX(corners.right_top.x, corners.right_bottom.x), left.rows));


	int dst_width = imageTransform1.cols;
	int dst_height = left.rows;
	Mat dst(dst_height, dst_width, CV_8UC3);
	dst.setTo(0);
	imageTransform1.copyTo(dst(Rect(0, 0, imageTransform1.cols, imageTransform1.rows)));

	left.copyTo(dst(Rect(0, 0, left.cols, left.rows)));
	OptimizeSeam(left, imageTransform1, dst);
	return dst;
}