import numpy as np


X = np.load('Data/train_image.npy', allow_pickle=True)
y = np.load('Data/train_label.npy', allow_pickle=True)


print(X.shape)
print(y.shape)
