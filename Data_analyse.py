import librosa
import os

index_range = {'train_demod': range(1, 6), 'train_record': range(11, 16),
               'test_demod': range(16, 21), 'test_record': range(6, 11)}
M = 10          # 10 segment for each record

for label in ['train_demod', 'train_record', 'test_demod', 'test_record']:
    max_data_len = 0
    path = 'data\\' + label
    if 'demod' in label:
        filetype = 'demod_'
    else:
        filetype = 'recorded_'
    for i, index in enumerate(index_range[label]):
        for seg_index in range(1, M+1):
            filename = filetype + str(index) + '.' + str(seg_index) + '.wav'
            file = os.path.join(path, filename)
            # print(file)
            data, fs = librosa.load(file, sr=None)
            if max_data_len < len(data):
                max_data_len = len(data)
                max_index = (index, seg_index)
            if i == 0 :
                fs_default = fs
            else:
                if fs != fs_default :
                    print("fs not identical for all sample in ", label)
                    break
    print(label, " max_data_len =", max_data_len, "max_index =", max_index, " fs_default =", fs_default)
