import cv2
from ImageProcessing_py.detection.detection_util import *

net = initYolo(cls_file="coco.names", model_conf="yolov4-tiny.cfg", model_weight="yolov4-tiny.weights")
cap = cv2.VideoCapture("right_11_12.mp4")

while True:
    hasFrame, frame = cap.read()

    if not hasFrame:
        break

    blob = cv2.dnn.blobFromImage(frame, 1 / 255, (inpWidth, inpHeight), [0, 0, 0], 1, crop=False)

    net.setInput(blob)

    outs = net.forward(getOutputsNames(net))

    postprocess(frame, outs)

    t, _ = net.getPerfProfile()
    label = 'Inference time: %.2f ms' % (t * 1000.0 / cv2.getTickFrequency())
    cv2.putText(frame, label, (0, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255))

    cv2.imshow("frame", frame)

    if cv2.waitKey(10) == 27:
        break
