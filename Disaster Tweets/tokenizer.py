import tensorflow as tf
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping
import pickle


def get_max_length(df):
    """
    get max token counts from train data,
    so we use this number as fixed length input to RNN cell
    """
    max_length = 0
    for row in df['clean_reviews']:
        if len(row.split(" ")) > max_length:
            max_length = len(row.split(" "))
    return max_length

df = pd.read_csv("Data/train_clean.csv")

max_length = get_max_length(df)

train_df, test_df = train_test_split(df, test_size=0.2)

train_reviews = train_df["clean_reviews"]
train_target = train_df["target"]

test_reviews = test_df["clean_reviews"]
test_target = test_df["target"]

tokenizer = Tokenizer(oov_token="<OOV>")
tokenizer.fit_on_texts(train_reviews)

with open("tokenizer.pickle", 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

vocab_size = len(tokenizer.word_index)

training_sequences = tokenizer.texts_to_sequences(train_reviews)
training_padded = pad_sequences(training_sequences, maxlen=max_length, padding="post")

testing_sequences = tokenizer.texts_to_sequences(test_reviews)
testing_padded = pad_sequences(testing_sequences, maxlen=max_length, padding="post")

embedding_dim = 250

model = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size+1, embedding_dim, input_length=max_length),
    tf.keras.layers.GlobalAveragePooling1D(),
    tf.keras.layers.Dense(64, activation="relu"),
    tf.keras.layers.Dense(1, activation="sigmoid")
])

model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

num_epochs = 50

es = EarlyStopping(monitor='val_loss', mode='min', verbose=1)

model.fit(training_padded, train_target, epochs=num_epochs,
          validation_data=(testing_padded, test_target), verbose=1, callbacks=[es])

model.save("tweets_classifier.h5")

# print(max_length)