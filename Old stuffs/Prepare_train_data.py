import os
from tqdm import tqdm
import cv2
import numpy as np
from random import shuffle

TRAIN_PATH = 'DogvsCat/Train'
IMG_SIZE = 100


def label_img(img):
    world_label = img.split('.')[0]
    if world_label == 'cat':
        return 0
    if world_label == 'dog':
        return 1


def create_train_data():
    training_data = []
    X = []
    y = []
    for img in tqdm(os.listdir(TRAIN_PATH)):
        label = label_img(img)
        path = os.path.join(TRAIN_PATH, img)
        img = cv2.resize(cv2.imread(path, cv2.IMREAD_GRAYSCALE), (IMG_SIZE, IMG_SIZE))

        training_data.append([img,label])

    shuffle(training_data)

    for features, label in training_data:
        X.append(features)
        y.append(label)

    X = np.array(X).reshape(-1, IMG_SIZE, IMG_SIZE, 1)

    np.save('DogvsCat/X_train.npy', X)
    np.save('DogvsCat/y_train.npy', y)

create_train_data()