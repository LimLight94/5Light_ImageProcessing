from __future__ import print_function
from PyQt5 import QtCore,QtWidgets
from PyQt5.QtWidgets import QLabel, QFileDialog, QApplication, QWidget, QHBoxLayout, QVBoxLayout, QPushButton
from ImageProcessing_py.stitching.panorama import Stitcher
from ImageProcessing_py.detection.detection_util import *
import sys
import threading
import imutils
import time


class MyApp(QWidget):

    def __init__(self):
        super().__init__()
        self.initUI()
        self.video1 = None
        self.video2 = None


    def initUI(self):

        n1 = QPushButton('1', self)
        n2 = QPushButton('2', self)
        self.video1_viewer_label = QLabel()
        self.video1_viewer_label.setGeometry(QtCore.QRect(10, 10, 400, 300))
        self.video2_viewer_label = QLabel()
        self.video2_viewer_label.setGeometry(QtCore.QRect(10, 10, 400, 300))
        self.label1 = QLabel()
        self.label2 = QLabel()
        n1.clicked.connect(self.pushButtonClicked_1)
        n2.clicked.connect(self.pushButtonClicked_2)
        stitch = QPushButton('정합(검출)')
        stitch.clicked.connect(self.pushButtonClicked_stitch)
        result = QPushButton('결과화면 보기')

        hbox = QHBoxLayout()
        hbox.addStretch(1)
        hbox.addWidget(n1)
        hbox.addStretch(2)
        hbox.addWidget(n2)
        hbox.addStretch(1)

        hbox_2 = QHBoxLayout()
        hbox_2.addWidget(self.label1)
        hbox_2.addWidget(self.label2)

        hbox_3 = QHBoxLayout()
        hbox_3.addWidget(stitch)
        hbox_3.addWidget(result)

        vbox = QVBoxLayout()
        vbox.addStretch(1)
        vbox.addLayout(hbox)
        vbox.addStretch(1)
        vbox.addLayout(hbox_2)
        vbox.addStretch(1)
        vbox.addLayout(hbox_3)

        self.setLayout(vbox)
        self.setGeometry(300, 300, 800, 400)
        self.setWindowTitle('Sample')
        self.show()

    def pushButtonClicked_1(self):

        fname = QFileDialog.getOpenFileName(self)

        self.label1.setText(fname[0])
        self.video1 = fname[0]


    def pushButtonClicked_2(self):

        fname = QFileDialog.getOpenFileName(self)

        self.label2.setText(fname[0])
        self.video2 = fname[0]


    def pushButtonClicked_stitch(self):
        leftStream = cv2.VideoCapture(self.video1)
        rightStream = cv2.VideoCapture(self.video2)
        if not leftStream:
            print("No files")
        else:
            print("Success a")

        if not rightStream:
            print("No files")
        else:
            print("Success b")
        time.sleep(2.0)

        # initialize the image stitcher, motion detector, and total
        stitcher = Stitcher()

        # write res
        stResult_type = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
        fps = leftStream.get(cv2.CAP_PROP_FPS)
        print("fps : ", fps)
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

            # stitching
            result = stitcher.stitch([left, right])

            # no homograpy could be computed
            if result is None:
                print("[INFO] homography could not be computed")
                break
            if stResult is None:
                stResult = cv2.VideoWriter("res/result.mp4", stResult_type, fps,
                                           (int(result.shape[1]), int(result.shape[0])), True)

            cv2.imshow("Left", left)
            cv2.imshow("Right", right)
            cv2.imshow("Result", result)
            cv2.waitKey(100)
            stResult.write(result)

        print("success")
        stResult.release()
        cv2.destroyAllWindows()






if __name__ == '__main__':

    app = QtWidgets.QApplication(sys.argv)

    ex = MyApp()
    sys.exit(app.exec())