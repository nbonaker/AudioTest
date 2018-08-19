import scipy.io.wavfile
import cmath
import math
from matplotlib import pyplot as plt
import winsound
import sys
import numpy as np
from PyQt5 import QtGui, QtCore, QtWidgets
import time

song_file = "Songs/youth.wav"
winsound.PlaySound(song_file, winsound.SND_ASYNC)


def dft(data):
    n = len(data)
    output = []
    for k in range(n):
        value = 0
        for t in range(n):
            value += data[t]*cmath.exp(-2j*cmath.pi*t*k/n)/len(data)
        output += [value]
    return output

def dft(data):
    return np.fft.fft(data)/len(data)

def boost(x):
    return math.pow(x, 0.8)

class AudioData:
    def __init__(self, data_file):
        self.data_file = data_file
        self.sample_rate, self.stereo_sample = scipy.io.wavfile.read(self.data_file)
        self.num_samples = len(self.stereo_sample)
        print("LOADED "+str(self.num_samples)+" SAMPLES")
        mono_sample = self.stereo_sample.max(axis=1)
        self.mono_sample = np.array([mono_sample, mono_sample]).T
        self.frequencies = []

    def write_wav(self):
        scipy.io.wavfile.write(self.data_file[:-4] + '_output.wav', self.sample_rate, self.mono_sample)

    def split(self, split_time=0.020):
        self.split_time = split_time
        self.split_length = int(split_time*self.sample_rate)
        print("SPLITTING @ "+str(self.split_length)+" SAMPLES/BIN . . .")
        self.bins_raw = [self.mono_sample.T[0][n*self.split_length:(n+1)*self.split_length] for n in range(0, self.num_samples//self.split_length)]
        print("CREATED "+str(len(self.bins_raw))+" BINS")

    def calc_fourier(self, bin):
        # print("CALCULATING FOURIER TRANSFORM . . .")
        if self.frequencies == []:
            self.frequencies = np.fft.fftfreq(self.split_length, d=self.split_time)
            self.frequencies *= self.split_length/2
        fourier = dft(self.bins_raw[bin])
        # print("CALCULATED FOURIER WITH "+str(len(self.frequencies)//2)+" FREQUENCIES FROM "+
        #       str(round(min(abs(self.frequencies[1:]))))+"Hz TO "+str(round(max(self.frequencies)))+"Hz")
        return fourier

    def calc_freq_dist(self, bin, fourier=[]):
        if fourier == []:
            fourier = self.calc_fourier(bin)
        amplitudes = [abs(c) for c in fourier]

        amplitudes = amplitudes[:len(amplitudes)//2+1]
        frequencies = self.frequencies
        energies = [(frequencies[i]**2)/1e8*(amplitudes[i]**2) for i in range(len(amplitudes))]
        scale_factor = np.sum(energies)
        energies = [boost(f/scale_factor) for f in energies]

        return energies

    def plot_freq_dist(self, start, stop):
        freq_dists = []
        for i in range(start, stop):
            self.calc_freq_dist(i)
        # frequencies = [self.frequencies[:len(self.frequencies) // 2 + 1]]
        # frequencies = [abs(f) for f in frequencies]
        #
        # a = np.array(freq_dists)
        # values = a.T
        # h, w = values.shape
        # fig, ax = plt.subplots(figsize=(9, 7))
        # # Make one larger so these values represent the edge of the data pixels.
        # y = np.array(frequencies).T
        # x = np.arange(0, self.split_time * w, self.split_time)
        #
        # pcm = ax.pcolormesh(x, y, values, rasterized=True, cmap='nipy_spectral')  # you don't need rasterized=True
        # fig.colorbar(pcm)
        # plt.show(block=True)

    def plot_fourier(self, bin, fourier=[]):
        if fourier == []:
            fourier = self.calc_fourier(bin)
        print("CALCULATING FOURIER SERIES . . .")

        def f(x):
            out = 0
            omega = -1 / np.pi
            for k in range(len(self.frequencies)):
                out += np.cos(x * self.frequencies[k]/omega) * fourier[k].real + np.sin(x * self.frequencies[k]/omega) * fourier[k].imag
            return out

        print("PLOTTING")
        plt.figure(figsize=(17, 8))
        x = np.arange(0, self.split_time, self.split_time/self.split_length)
        plt.plot(np.arange(0, self.split_time, self.split_time / self.split_length), self.bins_raw[bin])
        plt.plot(x, f(x), '.', color='purple')
        plt.show()


class BurningWidget(QtWidgets.QWidget):

    def __init__(self):
        super(BurningWidget, self).__init__()

        self.initUI()
        self.song_data = AudioData(song_file)
        self.song_data.split(0.02)

        self.start_time = time.time()
        self.bin_num = 0
        self.blocks = np.zeros((0, 0))
        self.block_w = 30


    def initUI(self):
        self.setGeometry(400, 200, 1100, 700)
        self.show()
        self.frame_timer = QtCore.QTimer()
        self.frame_timer.timeout.connect(self.check_frame)
        self.frame_timer.start(10)
        self.other = False

    def check_frame(self):
        self.other = (self.other == False)
        if time.time() - self.start_time >= self.song_data.split_time*(self.bin_num + 1):
            if self.other:
                self.get_bar()
            self.bin_num += 2
            self.repaint()

        if time.time() - self.start_time >= self.song_data.split_time*(self.bin_num + 2):
            print('lagging')

    def get_bar(self):
        self.add_bar(self.song_data.calc_freq_dist(self.bin_num)[:100])

    def add_bar(self, bar):
        bar = np.array([bar])
        bar = bar/np.max(bar)
        if self.blocks.size > 0:
            if self.blocks.shape[0] > 4:
                self.blocks = np.concatenate((bar, self.blocks[:-1]), axis=0)
            else:
                self.blocks = np.concatenate((bar, self.blocks), axis=0)
        else:
            self.blocks = bar

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
            h_ind = len(bar)
            for block in bar:
                index = block
                qp.setPen(QtGui.QColor(0, 0, index * 255))
                qp.setBrush(QtGui.QColor(0, 0, index * 255))
                qp.drawRect(w_ind*self.block_w, h_ind*h/len(bar), self.block_w, h/len(bar))
                h_ind -= 1
            w_ind += 1

app = QtWidgets.QApplication(sys.argv)
ex = BurningWidget()
sys.exit(app.exec_())

