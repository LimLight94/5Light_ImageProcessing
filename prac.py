import sys
from PyQt5.QtWidgets import QLabel, QFileDialog,QApplication, QWidget, QHBoxLayout, QVBoxLayout, QPushButton
from PyQt5.QtCore import Qt



class MyApp(QWidget):

        def __init__(self):
            super().__init__()
            self.initUI()

        def initUI(self):

          n1 = QPushButton('1',self)
          n2 = QPushButton('2', self)
          self.label=QLabel()
          n1.clicked.connect(self.pushButtonClicked)
          n2.clicked.connect(self.pushButtonClicked)
          stitch =QPushButton('정합(검출)')
          result = QPushButton('결과화면 보기')

          hbox = QHBoxLayout()
          hbox.addStretch(1)
          hbox.addWidget(n1)
          hbox.addStretch(2)
          hbox.addWidget(n2)
          hbox.addStretch(1)

          hbox_2 = QHBoxLayout()
          hbox_2.addWidget(self.label)

          hbox_3 = QHBoxLayout()
          hbox_3.addWidget(stitch)
          hbox_3.addWidget(result)


          vbox =QVBoxLayout()
          vbox.addStretch(1)
          vbox.addLayout(hbox)
          vbox.addStretch(1)
          vbox.addLayout(hbox_2)
          vbox.addStretch(1)
          vbox.addLayout(hbox_3)

          self.setLayout(vbox)
          self.setGeometry(300,300,300,200)
          self.setWindowTitle('Sample')
          self.show()
        def pushButtonClicked(self):
            fname = QFileDialog.getOpenFileName(self)
            self.label.setText(fname[0])

if __name__ == '__main__' :
        app = QApplication(sys.argv)
        ex = MyApp()
        sys.exit(app.exec())