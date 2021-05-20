import cv2

from ImageProcessing_py.detection.detection_util import *
from ImageProcessing_py.utils.restApi import sendEvent


class Detection:
    net = None
    frame_list = []
    tmp_list = []
    thumb_frame = None
    frame_idx = 0
    frame_idx_max = -1
    objects = []

    def __init__(self):
        self.net = initYolo(cls_file="ImageProcessing_py/detection/coco.names",
                            model_conf="ImageProcessing_py/detection/yolov4-tiny.cfg",
                            model_weight="ImageProcessing_py/detection/yolov4-tiny.weights")

    def detectObject(self, frame):
        self.frame_idx += 1
        blob = cv2.dnn.blobFromImage(frame, 1 / 255, (inpWidth, inpHeight), [0, 0, 0], 1, crop=False)
        self.net.setInput(blob)
        outs = self.net.forward(getOutputsNames(self.net))
        frame, deteceted, objects = postprocess(frame, outs)
        self.frame_list.append(frame)
        if deteceted and self.frame_idx_max == -1:
            self.objects = objects
            self.thumb_frame = frame
            self.frame_idx_max = self.frame_idx + 90

        if self.frame_idx == self.frame_idx_max:
            before = self.frame_idx_max - 180
            if before <= 0:
                before = 1
            self.tmp_list = self.frame_list[before:self.frame_idx_max]

            ##api
            sendEvent(1, self.objects, self.thumb_frame, self.tmp_list)

            ##초기화
            self.tmp_list.clear()
            self.frame_idx_max = -1
        return frame
