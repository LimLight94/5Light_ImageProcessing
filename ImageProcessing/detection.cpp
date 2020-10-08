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
	//net.setPreferableTarget(DNN_TARGET_OPENCL); OpenCL ��� ��

	Net net = readNetFromDarknet("yolov4-tiny.cfg", "yolov4-tiny.weights"); // ������ ���� ����
	net.setPreferableBackend(DNN_BACKEND_OPENCV);
	net.setPreferableTarget(DNN_TARGET_CPU);

	if (net.empty()) { // ��� �ε� ���� Ȯ��
		cerr << "Network load failed!" << endl;
	}

	ifstream fp("coco.names"); // �νĵǴ� ��ü �̸��� ���ִ� ����

	if (!fp.is_open()) { // ���� �ε� ���� Ȯ��   
		cerr << "Class file load failed!" << endl;
	}

	string name;

	while (!fp.eof()) { // ������ �о� classNames�� ����
		getline(fp, name);
		if (name.length())
			classNames.push_back(name);
	}

	fp.close();

	if (!cap.isOpened()) // ���� �ε� ���� Ȯ��
	{
		cerr << "Camera open failed!" << endl;
	}

	Mat frame;

	while (true)
	{
		cap >> frame; // ��ķ �������� frame�� ����

		if (frame.empty()) // frame�� �� �� break
			break;

		Mat inputBlob = blobFromImage(frame, 1 / 255.0, Size(inpWidth, inpHeight), Scalar(0, 0, 0), true, false); // frame�� blob���� ��ȯ

		net.setInput(inputBlob); // ��⿡ ����

		vector<Mat> outs;

		net.forward(outs, getOutputsNames(net)); // ��� ���̾� out�� �ֱ�

		postprocess(frame, outs);

		// ������ �ϳ� �� �ɸ� �ð� ���(���)
		vector<double> layersTimes;
		double freq = getTickFrequency() / 1000;
		double t = net.getPerfProfile(layersTimes) / freq; // ���̾� �ϳ��� �ɸ� �ð�
		string label = format("Inference time for a frame : %.2f ms", t);
		putText(frame, label, Point(0, 15), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 0, 255));

		Mat dst;

		resize(frame, dst, Size(1200, 800));

		imshow("frame", dst);

		if (waitKey(10) == 27) // ESC Ŭ�� �� ����
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

		vector<String> layersNames = net.getLayerNames(); // �Ű����(conv_0, relu_1 ~~ yolo_37) = 78��

		for (int i = 0; i < layersNames.size(); i++)
			cout << layersNames[i] << endl;

		names.resize(outLayers.size()); // 2���� ������ ����(yolo_30, yolo_37)

		for (size_t i = 0; i < outLayers.size(); ++i)
			names[i] = layersNames[outLayers[i] - 1];
	}

	return names; // yolo_30, yolo_37 ����
}

void postprocess(Mat& frame, const vector<Mat>& outs)
{
	vector<int> classIds; // ��ü
	vector<float> confidences; // �ŷڵ�
	vector<Rect> boxes; // �ڽ�

	// outs.size = 2, i = 0, �� 507, �� 85 ���͸� data�� ���, i = 1, �� 2028, �� 85 ���͸� data�� ����
	// ������ 1,2,3,4 index�� center_x, center_y, width, height, 5 index�� �ŷڵ�
	// 6~85������ �ش� ��ü�� Ȯ����!!

	for (size_t i = 0; i < outs.size(); ++i)
	{
		float* data = (float*)outs[i].data;

		for (int j = 0; j < outs[i].rows; ++j, data += outs[i].cols)
		{
			Mat scores = outs[i].row(j).colRange(5, outs[i].cols); // 6~85������ �ش� ��ü�� Ȯ��
			Point classIdPoint;
			double confidence;

			minMaxLoc(scores, 0, &confidence, 0, &classIdPoint); // ����� �ִ��� confidence �� �ְ� �ִ� ��ġ�� classIdPoint�� ����

			if (confidence > confThreshold) // �ŷڵ��� �־��� �ŷڵ����� ������ ��ġ�� �����ϰ� ��ü, �ŷڵ�, �ڽ��� ����
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
	//�����ִ� �ڽ� �� ���ڰ� ��ü�� Ȯ���� ���� ���� �ڽ��� ���ܵ�
	NMSBoxes(boxes, confidences, confThreshold, nmsThreshold, indices);

	// ��ü, �ŷڵ�, �ڽ��� ���� ȭ�鿡 ���� �ڽ��� �׸�
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