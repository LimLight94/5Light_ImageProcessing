#include "Stitching.h"
#include <iostream>


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