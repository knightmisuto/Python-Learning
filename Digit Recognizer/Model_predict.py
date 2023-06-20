import tensorflow as tf
import numpy as np
import pandas as pd

model = tf.keras.models.load_model('Digit recognizer.h5')

probability_model = tf.keras.Sequential([model, tf.keras.layers.Softmax()])

test_images = np.load('Data/test_image.npy')

Imageid = list(range(1,test_images.shape[0]+1))

predictions = probability_model.predict(test_images)

predicts = []

for i in range(test_images.shape[0]):
    predicts.append(np.argmax(predictions[i]))

data = {"Imageid": Imageid, 'Label': predicts}
data = pd.DataFrame(data)

data.to_csv("Data/my submission.csv", index=False)