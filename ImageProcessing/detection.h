#pragma once

#include "opencv2/opencv.hpp"
#include "opencv2/core/ocl.hpp"
#include <fstream>

using namespace cv;
using namespace cv::dnn;
using namespace std;

static float confThreshold = 0.5; // ½Å·Úµµ
static float nmsThreshold = 0.5;  // 
static int inpWidth = 416;
static int inpHeight = 416;
static vector<String> classNames;

void drawPred(int classId, float conf, int left, int top, int right, int bottom, Mat& frame);
void postprocess(Mat& frame, const vector<Mat>& outs);
vector<String> getOutputsNames(const Net& net);
void objectDetection(VideoCapture& cap);