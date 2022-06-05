import sys, random, os
from PyQt5.QtWidgets import QApplication, QWidget, QMainWindow, QToolBar, QLabel, QComboBox, QSpinBox, QVBoxLayout, \
    QHBoxLayout, QGridLayout, QAction, QToolButton, QColorDialog, QSplitter, QLineEdit, QTextEdit, QLayout
from PyQt5.QtGui import QPainter, QColor, QFont, QPalette, QPixmap, QPen, QImage, QIcon
from PyQt5.QtCore import Qt, QObject, pyqtSignal, QPoint
from models.MySimpleNet import MySimpleNet as Net
from run import evaluate
# from mnist import Net, evaluate
import torch
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

class DrawWidget(QWidget):
    def __init__(self):
        super(DrawWidget, self).__init__()
        self.initUI()

    def initUI(self):
        self.setGeometry(300, 300, 256, 256)
        self.setWindowTitle('Drawing')
        self.setAutoFillBackground(True)
        self.setPalette(QPalette(Qt.black))
        self.pix = QPixmap(self.size())

        self.pix.fill(Qt.black)
        # self.setMinimumSize(128, 128)
        self.setFixedSize(256, 256)
        self.label = QLabel()
        self.label.setPixmap(self.pix)

        self.layout = QVBoxLayout()
        self.layout.addWidget(self.label)

        self.start_pos = QPoint()
        self.end_pos = QPoint()
        self.init_style = Qt.SolidLine
        self.init_weight = 10
        self.init_color = QColor(Qt.white)

        self.setLayout(self.layout)
        self.show()

    def mousePressEvent(self, QMouseEvent):
        self.start_pos = QMouseEvent.pos()

    def mouseMoveEvent(self, QMouseEvent):
        painter = QPainter()
        pen = QPen()

        pen.setStyle(self.init_style)
        pen.setWidth(self.init_weight)
        pen.setColor(self.init_color)

        painter.begin(self.pix)
        painter.setPen(pen)
        painter.drawLine(self.start_pos, QMouseEvent.pos())
        painter.end()

        self.start_pos = QMouseEvent.pos()
        self.update()

    def paintEvent(self, QPaintEvent):
        painter = QPainter()
        painter.drawPixmap(QPoint(0, 0), self.pix)
        self.label.setPixmap(self.pix)

    def resizeEvent(self, QResizeEvent):
        if self.height() > self.pix.height() or self.width() > self.pix.width():
            new_pix = QPixmap(self.size())
            new_pix.fill(Qt.black)

            painter = QPainter(new_pix)
            painter.drawPixmap(QPoint(0, 0), self.pix)

            self.pix = new_pix

        QWidget.resizeEvent(self, QResizeEvent)

    def clear(self):
        clear_pix = QPixmap(self.size())
        clear_pix.fill(Qt.black)
        self.pix = clear_pix

        self.update()

    def set_style(self, s):
        self.init_style = s + 1

    def set_width(self, w):
        self.init_weight = w

    def set_color(self, c):
        self.init_color = c


class MAIN_Window(QMainWindow):
    def __init__(self):
        super().__init__()
        self.model = Net().cuda()
        self.model.load_state_dict(torch.load('trained_pts/MySimpleNet.pt'))
        self.initUI()

    def initUI(self):
        self.setGeometry(200, 200, 550, 300)
        self.setWindowTitle('MNIST DEMO')
        self.draw_widget = DrawWidget()
        widget = QWidget()

        self.setCentralWidget(widget)
        layout = QGridLayout()
        layout.setSpacing(40)

        layout.addWidget(self.draw_widget, 0, 0)

        self.text_edit = QTextEdit()
        self.text_edit.setFocusPolicy(Qt.NoFocus)

        lb = QLabel("识别结果: ")
        self.lbl = QLabel()

        self.createtoolbar()

        layout.addWidget(self.text_edit, 0, 1, 2, 1)

        splitter = QSplitter()
        layout.addWidget(splitter, 1, 0)
        splitter.addWidget(lb)
        splitter.addWidget(self.lbl)

        widget.setLayout(layout)
        self.setMinimumSize(300, 300)
        self.show()

    def createtoolbar(self):
        self.toolbar = QToolBar('Tool')

        self.style_label = QLabel('线条风格: ')
        self.style_combobox = QComboBox()
        self.style_combobox.addItem('Solid Line', Qt.SolidLine)
        self.style_combobox.addItem('DashLine', Qt.DashLine)
        self.style_combobox.addItem('DotLine', Qt.DotLine)
        self.style_combobox.addItem('DashDotLine', Qt.DashDotLine)
        self.style_combobox.addItem('DashDotDotLine', Qt.DashDotDotLine)
        self.style_combobox.currentIndexChanged.connect(self.draw_widget.set_style)

        self.widthLabel = QLabel('线条宽度: ')
        self.width_spin_box = QSpinBox()
        self.width_spin_box.setValue(10)
        self.width_spin_box.valueChanged.connect(self.set_width)

        self.colorbtn = QToolButton()
        self.colorbtn.setText('Color')
        pixmap = QPixmap(20, 20)
        pixmap.fill(Qt.black)
        self.colorbtn.setIcon(QIcon(pixmap))
        self.colorbtn.clicked.connect(self.show_color)

        self.clearbtn = QToolButton()
        self.clearbtn.setText('清空')
        self.clearbtn.clicked.connect(self.clear)

        self.calcbtn = QToolButton()
        self.calcbtn.setText('识别')
        self.calcbtn.clicked.connect(self.calc)

        self.toolbar.addWidget(self.style_label)
        self.toolbar.addWidget(self.style_combobox)
        self.toolbar.addSeparator()
        self.toolbar.addWidget(self.widthLabel)
        self.toolbar.addWidget(self.width_spin_box)
        self.toolbar.addSeparator()
        self.toolbar.addWidget(self.colorbtn)
        self.toolbar.addSeparator()
        self.toolbar.addWidget(self.clearbtn)
        self.toolbar.addSeparator()
        self.toolbar.addWidget(self.calcbtn)

        self.addToolBar(self.toolbar)

    def show_color(self):
        color = QColorDialog.getColor(Qt.black, self, '选择线条颜色')
        if color.isValid():
            self.draw_widget.set_color(color)
            pixmap = QPixmap(20, 20)
            pixmap.fill(color)
            self.colorbtn.setIcon(QIcon(pixmap))

    def clear(self):
        self.lbl.setText('')
        self.text_edit.setText('')
        self.draw_widget.clear()

    def set_width(self, w):
        self.draw_widget.set_width(w)

    # # reference : https://zhuanlan.zhihu.com/p/30120447
    # def judge_edge(self, img, length, flag, val=0):
    #     size = [-1, -1]
    #     for i in range(length):
    #         if flag == 'row':
    #             line1 = img[i, img[i, :] > val]
    #             line2 = img[length - 1 - i, img[length - 1 - i, :] > val]
    #         else:
    #             line1 = img[img[:, i] > val, i]
    #             line2 = img[img[:, length - i - 1] > val, length - 1 - i]
    #         if len(line1) > 0 and size[0] == -1:
    #             size[0] = i
    #         if len(line2) > 0 and size[1] == -1:
    #             size[1] = length - i - 1
    #         if size[0] != -1 and size[1] != -1:
    #             return size
    #     return [0, length]
    #
    # def crop(self, img):
    #     height = img.shape[0]
    #     width = img.shape[1]
    #     size = []
    #     size.append(self.judge_edge(img, height, 'row', val=0))
    #     size.append(self.judge_edge(img, width, 'column', val=0))
    #     return img[max(size[0][0]-10, 0):min(size[0][1] + 10, height),
    #            max(size[1][0]-10, 0):min(size[1][1] + 10, width)]

    def calc(self):
        self.draw_widget.pix.save('test.jpg', 'JPG')
        self.text_edit.setText('')

        img = Image.open('test.jpg')

        outut, pred = evaluate(self.model, img)
        self.lbl.setText(str(pred))
        outlist = sorted(outut.items(), key=lambda item: item[1], reverse=True)  # 根据概率排序
        for i in range(len(outlist)):
            self.text_edit.append('{}------{:.4f}'.format(outlist[i][0], outlist[i][1]))


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = MAIN_Window()
    ex.show()
    sys.exit(app.exec_())
