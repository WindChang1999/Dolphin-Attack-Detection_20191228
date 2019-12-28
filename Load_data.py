import xlrd
import numpy as np
def Load_data(Dataset):
    if Dataset == 'train' :
        path = r'train_dataset.xlsx'
    elif Dataset == 'test' :
        path = r'test_dataset.xlsx'
    elif Dataset == 'all' :
        path = r'all_dataset.xlsx'
    else :
        print('Error!')
    workbook = xlrd.open_workbook(path)
    sheet = workbook.sheet_by_index(0)
    data = []
    label = []
    for i in range(sheet.nrows) :
        label.append(sheet.row_values(i)[0])
        data.append(sheet.row_values(i)[1:])
    data = np.array(data)
    label = np.array(label)
    return data, label
