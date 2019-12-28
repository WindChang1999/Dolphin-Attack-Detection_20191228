from sklearn import svm
from sklearn.preprocessing import StandardScaler
from Load_data import Load_data
import numpy as np


if __name__ == '__main__':
    scaler = StandardScaler()
    data, label = Load_data('train')
    print(data.shape, label.shape)
    data = scaler.fit_transform(data)
    nmin = -3
    nmax = 3
    classifier = []

    C_range = np.logspace(-2, 10, 13)
    gamma_range = np.logspace(-12, 3, 16)
    for C in C_range :
        for gamma in gamma_range :
            CLF = svm.SVC(C=C, gamma=gamma)
            CLF.fit(data, label)
            classifier.append((C, gamma, CLF))

    testdata, label = Load_data('test')
    testdata = scaler.transform(testdata)

    best_pacc = 0
    best_nacc = 0
    best_acc = 0
    best_gamma = 0
    best_c = 0
    for C, gamma, clf in classifier :
        positive_cnt = 0
        negative_cnt = 0
        for input, type in zip(testdata, label):
            input = input.reshape(1, -1)
            pred = clf.predict(input)
            # print(input.shape)
            if pred == type and type == 1 :
                positive_cnt = positive_cnt + 1
            if pred == type and type == -1 :
                negative_cnt = negative_cnt + 1
            acc = (positive_cnt + negative_cnt) / len(label)
            pacc = 2 * positive_cnt / (len(label))
            nacc = 2 * negative_cnt / (len(label))
            if acc > best_acc :
                best_acc = acc
                best_gamma = gamma
                best_c = C
                best_pacc = pacc
                best_nacc = nacc

    print('*'*40)
    print("gamma = {}, C = {:f}\nTotal acc = {:.2%}".format(best_gamma, best_c, best_acc))
    print("Positive acc = {:.2%}".format(best_pacc))
    print("Negative acc = {:.2%}".format(best_nacc))

