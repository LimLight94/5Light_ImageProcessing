from __future__ import print_function
from ImageProcessing_py.stitching.basicmotiondetector import BasicMotionDetector
from ImageProcessing_py.stitching.panorama import Stitcher
import imutils
import time
from ImageProcessing_py.detection.detection_util import *


net = initYolo(cls_file="ImageProcessing_py/detection/coco.names", model_conf="ImageProcessing_py/detection/yolov4-tiny.cfg", model_weight="ImageProcessing_py/detection/yolov4-tiny.weights")
# initial stitching
leftStream = cv2.VideoCapture("res/left.mp4")
rightStream = cv2.VideoCapture("res/right.mp4")

time.sleep(2.0)

# initialize the image stitcher, motion detector, and total
stitcher = Stitcher()
motion = BasicMotionDetector(minArea=500)

# write res
stResult_type = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
fps = leftStream.get(cv2.CAP_PROP_FPS)
print("fps : ",fps)
# stResult = cv2.VideoWriter("stResult.mp4", stResult_type, fps, (int(800), int(225)), True)
stResult = None

# loop over frames from the res streams
while True:
    ret_l, left = leftStream.read()
    ret_r, right = rightStream.read()

    # resize the frames
    if left is None or right is None:
        break
    left = imutils.resize(left, width=400)
    right = imutils.resize(right, width=400)

    # stitch the frames together to form the panorama
    # IMPORTANT: you might have to change this line of code
    # depending on how your cameras are oriented; frames
    # should be supplied in left-to-right order
    result = stitcher.stitch([left, right])

    # detection
    blob = cv2.dnn.blobFromImage(image=result, scalefactor=1 / 255,
                                 size=(inpWidth, inpHeight), mean=[0, 0, 0], swapRB=1,
                                 crop=False)

    net.setInput(blob)

    outs = net.forward(getOutputsNames(net))

    postprocess(result, outs)

    t, _ = net.getPerfProfile()
    label = 'Inference time: %.2f ms' % (t * 1000.0 / cv2.getTickFrequency())
    cv2.putText(result, label, (0, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255))


    # no homograpy could be computed
    if result is None:
        print("[INFO] homography could not be computed")
        break
    if stResult is None :
        stResult = cv2.VideoWriter("res/result.mp4", stResult_type, fps, ( int(result.shape[1]), int(result.shape[0]) ), True)
    # # show the output images
    cv2.imshow("Result", result)
    cv2.imshow("Left Frame", left)
    cv2.imshow("Right Frame", right)
    stResult.write(result)
    key = cv2.waitKey(1) & 0xFF

    # if the `q` key was pressed, break from the loop
    if key == ord("q"):
        break

# do a bit of cleanup
stResult.release()
print("[INFO] cleaning up...")
cv2.destroyAllWindows()
