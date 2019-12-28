import os
import numpy as np
import xlsxwriter
import librosa

# def Maxpool(data, Width):
#     ret = []
#     for k in range(0, len(data), Width):
#         p = np.max(data[k: k+Width])
#         ret.append(p)
#     return np.array(ret)

index_range = {'train_demod': range(1, 6), 'train_record': range(11, 16),
               'test_demod': range(16, 21), 'test_record': range(6, 11)}
M = 10          # 10 segment for each record
MAX_LENGTH = 250000         # append zero to each sample to get uniform length sample
fs = 48000

train_workbook = xlsxwriter.Workbook(r'train_dataset.xlsx')
sheet1 = train_workbook.add_worksheet()
train_index = 0
test_workbook = xlsxwriter.Workbook(r'test_dataset.xlsx')
sheet2 = test_workbook.add_worksheet()
test_index = 0

# feature_bandwidth = [500, 1000]     # bandwidth use as feature
# nmin = int(feature_bandwidth[0] / fs * MAX_LENGTH)
# nmax = int(feature_bandwidth[1] / fs * MAX_LENGTH)

for label in ['train_demod', 'train_record', 'test_demod', 'test_record']:
    path = 'data\\' + label
    if 'demod' in label:
        filetype = 'demod_'
        data_type = +1              # demod denote as +1
    else:
        filetype = 'recorded_'
        data_type = -1              # record denote as -1

    for i, index in enumerate(index_range[label]):
        for seg_index in range(1, M+1):
            filename = filetype + str(index) + '.' + str(seg_index) + '.wav'
            file = os.path.join(path, filename)
            data, fs = librosa.load(file, sr=None)

            # ---- use fft as feature ----
            # Zero = np.zeros(MAX_LENGTH-len(data))
            # data = np.hstack((data, Zero))
            # # calculate fft
            # fft = np.fft.fft(data)[0:len(data) // 2]
            # feature = np.array([abs(fft[n]) for n in range(nmin, nmax + 1)])
            # feature = Maxpool(feature, 50)

            # ---- use aver_mfcc as feature ----
            mfccs = librosa.feature.mfcc(y=data, sr=fs, n_mfcc=40)
            aver_mfcc = np.mean(mfccs, axis=1)
            feature = aver_mfcc

            if 'train' in label:
                sheet1.write(train_index, 0, data_type)
                for j, f in enumerate(feature):
                    sheet1.write(train_index, j+1, f)
                train_index += 1
            else:
                sheet2.write(test_index, 0, data_type)
                for j, f in enumerate(feature):
                    sheet2.write(test_index, j+1, f)
                test_index += 1
train_workbook.close()
test_workbook.close()
print("Pretreatment complete")



