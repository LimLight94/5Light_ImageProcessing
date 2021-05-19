import cv2

from ImageProcessing_py.detection.detection_util import *


class Detection:
    net = None

    def __init__(self):
        self.net = initYolo(cls_file="ImageProcessing_py/detection/coco.names", model_conf="ImageProcessing_py/detection/yolov4-tiny.cfg", model_weight="ImageProcessing_py/detection/yolov4-tiny.weights")

    def detectObject(self, frame):
        blob = cv2.dnn.blobFromImage(frame, 1 / 255, (inpWidth, inpHeight), [0, 0, 0], 1, crop=False)
        self.net.setInput(blob)
        outs = self.net.forward(getOutputsNames(self.net))
        frame = postprocess(frame, outs)
        return frame
