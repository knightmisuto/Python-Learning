import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

train_data = np.load("Data/train_data.npy", allow_pickle=True)
val_data = np.load("Data/val_data.npy", allow_pickle=True)
train_labels = np.load("Data/train_labels.npy", allow_pickle=True)
val_labels = np.load("Data/val_labels.npy", allow_pickle=True)

file = open("num_vocab.txt", "r")
vocab_size = file.read()
vocab_size = int(vocab_size) + 1

model = keras.Sequential([
    layers.Embedding(vocab_size, 16),
    layers.Dropout(0.2),
    layers.GlobalAveragePooling1D(),
    layers.Dropout(0.2),
    layers.Dense(16, activation='relu'),
    layers.Dropout(0.2),
    layers.Dense(3),
])

model.compile(optimizer='adam',
              loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

epoch = 300
early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)

model.fit(train_data, train_labels, epochs=epoch, validation_data=(val_data, val_labels), verbose=2)