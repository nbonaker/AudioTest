import sys
import numpy as np
from PyQt5 import QtGui, QtCore, QtWidgets


class BurningWidget(QtWidgets.QWidget):

    def __init__(self):
        super(BurningWidget, self).__init__()

        self.initUI()
        self.blocks = np.zeros((0, 0))
        self.block_w = 10


    def initUI(self):
        self.setGeometry(400, 200, 1100, 700)
        self.show()

    def add_bar(self, bar):
        bar = np.array([bar])
        bar = bar/np.max(bar)
        if self.blocks.size > 0:
            self.blocks = np.concatenate((self.blocks, bar), axis=0)
        else:
            self.blocks = bar
        print(self.blocks)

    def paintEvent(self, e):

        qp = QtGui.QPainter()
        qp.begin(self)
        self.drawWidget(qp)
        qp.end()

    def drawWidget(self, qp):
        w = self.geometry().width()
        h = self.geometry().height()
        w_ind = 0
        for bar in self.blocks:
            h_ind = 0
            for block in bar:
                index = block
                qp.setPen(QtGui.QColor(0, 0, index * 255))
                qp.setBrush(QtGui.QColor(0, 0, index * 255))
                qp.drawRect(w_ind*self.block_w, h_ind*h/len(bar), self.block_w, h/len(bar))
                h_ind += 1
            w_ind += 1


def main():
    app = QtWidgets.QApplication(sys.argv)
    ex = BurningWidget()
    for i in range(12):
        ex.add_bar(np.random.random_integers(0,15,100))
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
