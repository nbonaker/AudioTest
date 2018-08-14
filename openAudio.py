import scipy.io.wavfile
import numpy as np
import cmath
from matplotlib import pyplot as plt


def dft(data):
    n = len(data)
    output = []
    for k in range(n):
        value = 0
        for t in range(n):
            value += data[t]*cmath.exp(-2j*cmath.pi*t*k/n)
        output += [value]
    return output


class AudioData:
    def __init__(self, data_file):
        self.data_file = data_file
        self.sample_rate, self.stereo_sample = scipy.io.wavfile.read(self.data_file)
        self.num_samples = len(self.stereo_sample)
        print("LOADED "+str(self.num_samples)+" SAMPLES")
        mono_sample = self.stereo_sample.max(axis=1)
        self.mono_sample = np.array([mono_sample, mono_sample]).T

    def write_wav(self):
        scipy.io.wavfile.write(self.data_file[:-4] + '_output.wav', self.sample_rate, self.mono_sample)

    def split(self, split_time=0.020):
        self.split_length = int(split_time*self.sample_rate)
        print("SPLITTING @ "+str(self.split_length)+" SAMPLES/BIN")
        self.bins_raw = [self.mono_sample.T[0][n*self.split_length:(n+1)*self.split_length] for n in range(0, self.num_samples//self.split_length)]
        print("CREATED "+str(len(self.bins_raw))+" BINS")


song_data = AudioData("Songs/YOUTH.wav")
song_data.split()
print(dft(song_data.bins_raw[1000]))

# plt.plot(dft(song_data.bins_raw[1000]))
# plt.show()
