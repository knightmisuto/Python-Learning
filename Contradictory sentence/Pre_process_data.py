import numpy as np
import pandas as pd
import pickle
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

datasets = pd.read_csv("Data/train.csv")

training_size = int(len(datasets)*0.8)

premise = datasets.premise.values
hypothesis = datasets.hypothesis.values
labels = datasets.label.values

train_premise = premise[:training_size]
val_premise = premise[training_size:]
train_hypothesis = hypothesis[:training_size]
val_hypothesis = hypothesis[training_size:]
train_labels = labels[:training_size]
val_labels = labels[training_size:]

tokenizer = Tokenizer(oov_token="<OOV>")
tokenizer.fit_on_texts(train_premise)
tokenizer.fit_on_texts(train_hypothesis)
word_index = tokenizer.word_index

file = open("num_vocab.txt", "w+")
file.write(str(len(word_index)))
file.close()

with open('tokenizer.pickle', 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

train_premise = tokenizer.texts_to_sequences(train_premise)
train_hypothesis = tokenizer.texts_to_sequences(train_hypothesis)
val_premise = tokenizer.texts_to_sequences(val_premise)
val_hypothesis = tokenizer.texts_to_sequences(val_hypothesis)

padded_train_premise = pad_sequences(train_premise, padding='post', truncating='post')
padded_train_hypothesis = pad_sequences(train_hypothesis, padding='post', truncating='post')
padded_val_premise = pad_sequences(val_premise, padding='post', maxlen=padded_train_premise.shape[1], truncating='post')
padded_val_hypothesis = pad_sequences(val_hypothesis, padding='post', maxlen=padded_train_hypothesis.shape[1], truncating='post')

train_datasets = tf.keras.layers.concatenate([padded_train_premise, padded_train_hypothesis], axis=1)
val_datasets = tf.keras.layers.concatenate([padded_val_premise, padded_val_hypothesis], axis=1)

np.save("Data/train_data", train_datasets)
np.save("Data/val_data", val_datasets)
np.save("Data/train_labels", train_labels)
np.save("Data/val_labels", val_labels)