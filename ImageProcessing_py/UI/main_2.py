from __future__ import print_function
from PyQt5 import QtCore,QtGui,QtWidgets
import sys

from PyQt5.QtCore import QUrl
from PyQt5.QtWidgets import QRadioButton,QGroupBox, QFileDialog, QApplication,QCheckBox, QWidget, QHBoxLayout, QVBoxLayout, QPushButton
from PyQt5.QtMultimedia import QMediaContent, QMediaPlayer
from PyQt5.QtGui import QPixmap, QImage
from time import sleep
import threading
from ImageProcessing_py.stitching.basicmotiondetector import BasicMotionDetector
from ImageProcessing_py.stitching.panorama import Stitcher
import imutils
import time
from ImageProcessing_py.detection.detection_util import *


class MyApp(QWidget):

    def __init__(self):
        super().__init__()
        self.initUI()
        self.video1 = None
        self.video2 = None




    def initUI(self):
        self.vbox = QVBoxLayout()
        self.hbox=  QHBoxLayout()


        self.button_1 =QPushButton("Vieo_1",self)
        self.button_1.resize(200,100)

        self.button_2 =QPushButton("Video_2",self)
        self.button_2.resize(200,100)
        self.button_3 =QPushButton("Select Video",self)
        self.button_3.resize(200,150)
        self.button_1.setVisible(0)
        self.button_2.setVisible(0)
        self.button_3.setVisible(0)
        self.button_4 =QPushButton("실행",self)

        groupBox= QGroupBox("Mode",self)
        groupBox.move(40,300)
        groupBox.resize(200,200)

        groupBox_2 = QGroupBox("Attribute", self)
        groupBox_2.move(300, 300)
        groupBox_2.resize(200, 200)

        self.radio1 = QRadioButton("Stitching", self)
        self.radio1.move(50,350)
        self.radio1.clicked.connect(self.radioButtonClicked)
        self.radio2 = QRadioButton("Panorama", self)
        self.radio2.move(50,400)
        self.radio2.clicked.connect(self.radioButtonClicked)

        self.check1 =QCheckBox("Detection",self)
        self.check1.move(310,350)
        self.check2 = QCheckBox("Blur", self)
        self.check2.move(310, 400)

        self.button_4.move(600,300)
        self.button_4.resize(200,200)


        self.setGeometry(300, 300, 900, 1000)
        self.setWindowTitle('Sample')
        self.show()

    def radioButtonClicked(self):

        if self.radio1.isChecked():

            self.button_1.setVisible(1)
            self.button_2.setVisible(1)
            self.button_3.setVisible(0)
            self.button_1.move(200,50)
            self.button_2.move(450,50)

        elif self.radio2.isChecked():
            self.button_1.setVisible(0)
            self.button_2.setVisible(0)
            self.button_3.setVisible(1)
            self.button_3.move(300,50)




if __name__ == '__main__':

    app = QtWidgets.QApplication(sys.argv)

    ex = MyApp()
    sys.exit(app.exec())
