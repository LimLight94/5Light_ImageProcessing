import sys
import numpy as np
import cv2

def blurring(frame):
    model = 'res10_300x300_ssd_iter_140000_fp16.caffemodel'
    config = 'deploy.prototxt'
    #model = 'opencv_face_detector_uint8.pb'
    #config = 'opencv_face_detector.pbtxt'

    net = cv2.dnn.readNet(model, config)

    if net.empty():
        print('Net open failed!')

    blob = cv2.dnn.blobFromImage(frame, 1, (300, 300), (104, 177, 123))
    net.setInput(blob)
    detect = net.forward()

    (h, w) = frame.shape[:2]
    detect = detect[0, 0, :, :]

    for i in range(detect.shape[0]):
        confidence = detect[i, 2]
        if confidence < 0.2:
            break

        x1 = int(detect[i, 3] * w)
        y1 = int(detect[i, 4] * h)
        x2 = int(detect[i, 5] * w)
        y2 = int(detect[i, 6] * h)

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0))
        face = frame[y1:y2, x1:x2]
        blur = cv2.blur(face, (17, 17))
        frame[y1:y2, x1:x2] = blur
        label = 'Face: %4.3f' % confidence
        cv2.putText(frame, label, (x1, y1 - 1), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 1, cv2.LINE_AA)

    return frame
