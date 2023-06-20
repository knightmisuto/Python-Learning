import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.preprocessing import LabelEncoder

data = pd.read_csv("House price data/train.csv")

data_train_num = data.select_dtypes(exclude=['object'])
data_train_num.fillna(0, inplace=True)

data_train_cat = data.select_dtypes(include=['object'])
data_train_cat.fillna('NONE', inplace=True)

col_train_cat = list(data_train_cat.columns)

enc = LabelEncoder()

for col in col_train_cat:
    data_train_cat[col] = data_train_cat[col].astype('str')
    data_train_cat[col] = enc.fit_transform(data_train_cat[col])

data = data_train_num.merge(data_train_cat, left_index=True, right_index=True)
data = pd.get_dummies(data, prefix='', prefix_sep='')
data.pop("Id")

train_dataset = data.sample(frac=0.8, random_state=0)
test_dataset = data.drop(train_dataset.index)

train_stats = train_dataset.describe()
train_stats.pop("SalePrice")
train_stats = train_stats.transpose()

train_labels = train_dataset.pop("SalePrice")
test_labels = test_dataset.pop("SalePrice")

def norm(x):
  return (x - train_stats['mean']) / train_stats['std']

normed_train_data = norm(train_dataset)
normed_test_data = norm(test_dataset)

def build_model():
    model = keras.Sequential([layers.Dense(512, activation='relu', input_shape=[len(train_dataset.keys())]),
                              layers.Dense(512, activation='relu'),
                              layers.Dense(1)
                              ])

    optimizer = tf.keras.optimizers.Adam()

    model.compile(loss='mse',
                  optimizer=optimizer,
                  metrics=['mae', 'mse'])
    return model

model = build_model()

early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)

class haltCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if(logs.get('val_loss') <= 0.02):
            print("\n\n\nReached 0.05 loss value so cancelling training!\n\n\n")
            self.model.stop_training = True

trainingStopCallback = haltCallback()

EPOCHS = 1000

history = model.fit(
    normed_train_data, train_labels,
    epochs=EPOCHS, validation_split=0.2, verbose=1,
    callbacks=[trainingStopCallback])

model.save('House_price_model.h5')

test_predictions = model.predict(normed_test_data).flatten()

a = plt.axes(aspect='equal')
plt.scatter(test_labels, test_predictions)
plt.xlabel('True Values [price]')
plt.ylabel('Predictions [price]')
lims = [0, 800000]
plt.xlim(lims)
plt.ylim(lims)
plt.plot(lims, lims)
plt.show()