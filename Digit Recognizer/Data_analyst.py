import numpy as np
import pandas as pd

def modified_train_data():
    labels = []
    features = []
    data = pd.read_csv("Data/train.csv")
    for row_num in range(data.shape[0]):
        row = list(data.loc[row_num, :])
        label = row[0]
        row.pop(0)
        feature = np.array(row, dtype=np.uint8)
        feature = feature.reshape((28, 28))
        labels.append(label)
        features.append(feature)
        print("{}% done.".format(round((row_num/(data.shape[0]+1))*100, 2)))
    features = np.stack(features)
    np.save('Data/train_label', labels)
    np.save('Data/train_image', features)

def modified_test_data():
    features = []
    data = pd.read_csv("Data/test.csv")
    for row_num in range(data.shape[0]):
        row = list(data.loc[row_num, :])
        feature = np.array(row, dtype=np.uint8)
        feature = feature.reshape((28,28))
        features.append(feature)
        print("{}% done.".format(round((row_num/(data.shape[0]+1))*100, 2)))
    features = np.stack(features)
    np.save('Data/test_image', features)

modified_test_data()