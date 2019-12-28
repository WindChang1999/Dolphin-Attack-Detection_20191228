from sklearn import svm
from Load_data import Load_data
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
traindata, trainlabel = Load_data('train')
traindata = scaler.fit_transform(traindata)

CLF = svm.SVC(C=0.1, gamma=0.001, kernel='linear')
CLF.fit(traindata, trainlabel)

testdata, testlabel = Load_data('test')
testdata = scaler.transform(testdata)


for data, label in [(traindata, trainlabel), (testdata, testlabel)] :
    TP, FN, FP, TN = (0, 0, 0, 0)
    for input, type in zip(data, label):
        input = input.reshape(1, -1)
        pred = CLF.predict(input)
        # print(input.shape)
        if pred == type :
            if type == 1 :
                TP += 1
            else :
                TN += 1
        else :
            if type == 1 :
                FN += 1
            else :
                FP += 1
    print("positive acc = {:.2%}".format(TP / (TP + FN)))
    print("negative acc = {:.2%}".format(TN / (FP + TN)))
    print("Total acc = {:.2%}".format((TP + TN) / (TP + FN + FP + TN)))

