import scipy.io.wavfile
import numpy as np
import cmath
import math
from matplotlib import pyplot as plt


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
    return math.pow(x, 0.66)

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
        print("CALCULATING FOURIER TRANSFORM . . .")
        if self.frequencies == []:
            self.frequencies = np.fft.fftfreq(self.split_length, d=song_data.split_time)
            self.frequencies *= self.split_length/2
        fourier = dft(self.bins_raw[bin])
        print("CALCULATED FOURIER WITH "+str(len(self.frequencies)//2)+" FREQUENCIES FROM "+
              str(round(min(abs(self.frequencies[1:]))))+"Hz TO "+str(round(max(self.frequencies)))+"Hz")
        return fourier

    def calc_freq_dist(self, bin, fourier=[]):
        if fourier == []:
            fourier = self.calc_fourier(bin)
        fourier = [abs(c) for c in fourier]
        fourier = [f/max(fourier) for f in fourier]
        fourier = [boost(f) for f in fourier]
        fourier = [fourier[:len(fourier)//2+1]]
        return fourier[0]

    def plot_freq_dist(self, start, stop):
        freq_dists=[]
        for i in range(start, stop):
            freq_dists += [self.calc_freq_dist(i)]
        frequencies = [self.frequencies[:len(self.frequencies) // 2 + 1]]
        frequencies = [abs(f) for f in frequencies]

        a = np.array(freq_dists)
        values = a.T
        h, w = values.shape
        fig, ax = plt.subplots(figsize=(9, 7))
        # Make one larger so these values represent the edge of the data pixels.
        y = np.array(frequencies).T
        x = np.arange(0, self.split_time * (w), self.split_time)

        pcm = ax.pcolormesh(x, y, values, rasterized=True)  # you don't need rasterized=True
        fig.colorbar(pcm)
        plt.show()

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
        plt.plot(np.arange(0, self.split_time, self.split_time / self.split_length), song_data.bins_raw[bin])
        plt.plot(x, f(x), '.', color='purple')
        plt.show()


song_data = AudioData("Songs/YOUTH.wav")
song_data.split()
index = 3
ft = song_data.calc_fourier(index)
song_data.plot_freq_dist(1000,2000)