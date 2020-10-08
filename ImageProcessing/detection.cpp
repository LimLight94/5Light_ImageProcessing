#include "Detection.h"

void objectDetection(VideoCapture& cap)
{
	//ocl::Context context;
	//if (!context.create(ocl::Device::TYPE_GPU))
	//{
	//	cout << "Context not created!" << endl;
	//	return -1;
	//}

	//ocl::Device(context.device(0));

	//ocl::setUseOpenCL(true);
	//net.setPreferableTarget(DNN_TARGET_OPENCL); OpenCL 사용 시

	Net net = readNetFromDarknet("yolov4-tiny.cfg", "yolov4-tiny.weights"); // 딥러닝 파일 적용
	net.setPreferableBackend(DNN_BACKEND_OPENCV);
	net.setPreferableTarget(DNN_TARGET_CPU);

	if (net.empty()) { // 모듈 로드 실패 확인
		cerr << "Network load failed!" << endl;
	}

	ifstream fp("coco.names"); // 인식되는 객체 이름이 모여있는 파일

	if (!fp.is_open()) { // 파일 로드 실패 확인   
		cerr << "Class file load failed!" << endl;
	}

	string name;

	while (!fp.eof()) { // 파일을 읽어 classNames에 저장
		getline(fp, name);
		if (name.length())
			classNames.push_back(name);
	}

	fp.close();

	if (!cap.isOpened()) // 비디오 로드 실패 확인
	{
		cerr << "Camera open failed!" << endl;
	}

	Mat frame;

	while (true)
	{
		cap >> frame; // 웹캠 프레임을 frame에 넣음

		if (frame.empty()) // frame이 빌 시 break
			break;

		Mat inputBlob = blobFromImage(frame, 1 / 255.0, Size(inpWidth, inpHeight), Scalar(0, 0, 0), true, false); // frame을 blob으로 변환

		net.setInput(inputBlob); // 모듈에 넣음

		vector<Mat> outs;

		net.forward(outs, getOutputsNames(net)); // 출력 레이어 out에 넣기

		postprocess(frame, outs);

		// 프레임 하나 당 걸린 시간 출력(상단)
		vector<double> layersTimes;
		double freq = getTickFrequency() / 1000;
		double t = net.getPerfProfile(layersTimes) / freq; // 레이어 하나에 걸린 시간
		string label = format("Inference time for a frame : %.2f ms", t);
		putText(frame, label, Point(0, 15), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 0, 255));

		Mat dst;

		resize(frame, dst, Size(1200, 800));

		imshow("frame", dst);

		if (waitKey(10) == 27) // ESC 클릭 시 종료
			break;
	}
}

vector<String> getOutputsNames(const Net& net)
{
	static vector <String> names;

	if (names.empty())
	{
		vector<int> outLayers = net.getUnconnectedOutLayers(); // 66, 78

		for (int i = 0; i < outLayers.size(); i++)
			cout << outLayers[i] << endl;

		vector<String> layersNames = net.getLayerNames(); // 신경망들(conv_0, relu_1 ~~ yolo_37) = 78개

		for (int i = 0; i < layersNames.size(); i++)
			cout << layersNames[i] << endl;

		names.resize(outLayers.size()); // 2개로 사이즈 조정(yolo_30, yolo_37)

		for (size_t i = 0; i < outLayers.size(); ++i)
			names[i] = layersNames[outLayers[i] - 1];
	}

	return names; // yolo_30, yolo_37 리턴
}

void postprocess(Mat& frame, const vector<Mat>& outs)
{
	vector<int> classIds; // 객체
	vector<float> confidences; // 신뢰도
	vector<Rect> boxes; // 박스

	// outs.size = 2, i = 0, 행 507, 열 85 벡터를 data에 담고, i = 1, 행 2028, 열 85 벡터를 data에 담음
	// 벡터의 1,2,3,4 index는 center_x, center_y, width, height, 5 index는 신뢰도
	// 6~85까지는 해당 물체일 확률임!!

	for (size_t i = 0; i < outs.size(); ++i)
	{
		float* data = (float*)outs[i].data;

		for (int j = 0; j < outs[i].rows; ++j, data += outs[i].cols)
		{
			Mat scores = outs[i].row(j).colRange(5, outs[i].cols); // 6~85까지의 해당 물체일 확률
			Point classIdPoint;
			double confidence;

			minMaxLoc(scores, 0, &confidence, 0, &classIdPoint); // 행렬의 최댓값을 confidence 에 넣고 최댓값 위치를 classIdPoint에 넣음

			if (confidence > confThreshold) // 신뢰도가 주어진 신뢰도보다 높으면 위치를 설정하고 객체, 신뢰도, 박스에 넣음
			{
				int centerX = (int)(data[0] * frame.cols);
				int centerY = (int)(data[1] * frame.rows);
				int width = (int)(data[2] * frame.cols);
				int height = (int)(data[3] * frame.rows);
				int left = centerX - width / 2;
				int top = centerY - height / 2;

				classIds.push_back(classIdPoint.x);
				confidences.push_back((float)confidence);
				boxes.push_back(Rect(left, top, width, height));
			}
		}
	}

	vector<int> indices;
	//겹쳐있는 박스 중 상자가 물체일 확률이 가장 높은 박스만 남겨둠
	NMSBoxes(boxes, confidences, confThreshold, nmsThreshold, indices);

	// 객체, 신뢰도, 박스를 토대로 화면에 예측 박스를 그림
	for (size_t i = 0; i < indices.size(); ++i)
	{
		int idx = indices[i];
		Rect box = boxes[idx];
		drawPred(classIds[idx], confidences[idx], box.x, box.y, box.x + box.width, box.y + box.height, frame);
	}
}

void drawPred(int classId, float conf, int left, int top, int right, int bottom, Mat& frame)
{
	rectangle(frame, Point(left, top), Point(right, bottom), Scalar(0, 0, 255));
	string label = format("%.2f", conf);
	if (!classNames.empty())
	{
		CV_Assert(classId < (int)classNames.size());
		label = classNames[classId] + ": " + label;
	}

	int baseLine;
	Size labelSize = getTextSize(label, FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
	top = max(top, labelSize.height);

	putText(frame, label, Point(left, top), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(255, 255, 255));
}