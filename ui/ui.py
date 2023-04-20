# PyQt UI 界面
import sys

import cv2
import numpy as np
import torch
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QVBoxLayout, QHBoxLayout

from ui.form import Ui_MainWindow
from capture.image_get import Capture

img = None

ww = Ui_MainWindow()


# test_net: model.Net = torch.load("output/train_net_9.pth")
# test_net.to(model.device)

class MainWindow(QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        self.resize(800, 640)
        self.load_ui()
        self.capture = None

    def load_ui(self):
        self.form = None
        # loader = QUiLoader()
        # path = os.fspath(Path(__file__).resolve().parent / "form.ui")
        # ui_file = QFile(path)
        # ui_file.open(QFile.ReadOnly)
        # q_widget = loader.load(ui_file, self)
        # ui_file.close()
        # self.ui = q_widget
        self.button1 = QPushButton('按键1', self)
        self.button1.clicked.connect(lambda: self.clickButton())
        self.qh_box_layout = QHBoxLayout(self)
        qv_box_layout = QVBoxLayout(self)
        # self.qh_box_layout.addChildLayout(qv_box_layout)
        # QPushButton('按键2', parent=self.qh_box_layout)
        self.init_ui()

    def clickButton(self):
        print(self.button1.text() + '被点击')

    def buttonClick(self):
        sender = self.sender()
        print(sender.text(), "被点击了")

    def init_ui(self):
        # ww.pushButton_13.clicked.connect(self.button_13_clicked())
        pass

    def button_13_clicked(self):
        print(111)
        if self.capture is None:
            self.capture = Capture()
            self.capture.open_capture()
        if self.capture.capture.isOpened():
            self.capture.get_a_frame()
            ww.open2show()


app = QApplication([])
widget = MainWindow()

ww.setupUi(widget)
widget.form = ww
widget.init_ui()
widget.show()
sys.exit(app.exec_())


