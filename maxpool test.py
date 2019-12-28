import numpy as np

def Maxpool(data, Width):
    ret = []
    for k in range(0, len(data), Width):
        p = np.max(data[k: k+Width])
        ret.append(p)
    return np.array(ret)

a = np.array([1, 2, 3, 4, 5, 6, 7, 8])

print(Maxpool(a, 9))