import librosa
from librosa import display
import os
import numpy as np
import matplotlib.pyplot as plt

path = 'data\\train_demod'
filename = 'demod_4.2.wav'
file_path = os.path.join(path, filename)
# print(file)

y, sr = librosa.load(file_path, sr=None)

mfccs = librosa.feature.mfcc(y=y, sr=sr)

# fdata = fs/len(data) * np.arange(0, len(data)//2)
# fft = np.fft.fft(data)[0:len(data)//2]
# print("fft len =", len(fft))
# plt.subplot(2, 1, 1)
# plt.plot(fdata, abs(fft))

aver_mfccs = np.mean(mfccs, axis=1)

plt.figure()
display.specshow(mfccs, x_axis='time', y_axis='mel', sr=sr)
plt.colorbar()
plt.figure()
plt.plot(aver_mfccs)
plt.show()