import tensorflow as tf
from tensorflow.keras.layers import Dense, LSTM
from tensorflow.keras.models import Model
from tensorflow.keras.models import Sequential
import numpy as np
import pandas as pd
import tensorflow_hub as hub
from sklearn.model_selection import train_test_split

embed = hub.KerasLayer('Word2Vec/Wiki-words-250-with-normalization_2')


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


def get_word2vec_enc(reviews):
    """
    get word2vec value for each word in sentence.
    concatenate word in numpy array, so we can use it as RNN input
    """
    encoded_reviews = []
    for review in reviews:
        tokens = review.split(" ")
        word2vec_embedding = embed(tokens)
        encoded_reviews.append(word2vec_embedding)
    return encoded_reviews


def get_padded_encoded_reviews(encoded_reviews):
    """
    for short sentences, we prepend zero padding so all input to RNN has same length
    """
    padded_reviews_encoding = []
    for enc_review in encoded_reviews:
        zero_padding_cnt = max_length - enc_review.shape[0]
        pad = np.zeros((1, 250))
        for i in range(zero_padding_cnt):
            enc_review = np.concatenate((pad, enc_review), axis=0)
        padded_reviews_encoding.append(enc_review)
    return padded_reviews_encoding


def sentiment_encode(sentiment):
    """
    return one hot encoding for Y value
    """
    if sentiment == 1:
        return [1, 0]
    else:
        return [0, 1]


def preprocess(df):
    """
    encode text value to numeric value
    """
    # encode words into word2vec
    reviews = df['clean_reviews'].tolist()

    encoded_reviews = get_word2vec_enc(reviews)
    padded_encoded_reviews = get_padded_encoded_reviews(encoded_reviews)
    # encoded sentiment
    sentiments = df['target'].tolist()
    encoded_sentiment = [sentiment_encode(sentiment) for sentiment in sentiments]
    X = np.array(padded_encoded_reviews)
    Y = np.array(encoded_sentiment)
    return X, Y

df = pd.read_csv("Data/train_clean.csv")
train_df, test_df = train_test_split(df, test_size=0.2)

max_length = get_max_length(train_df)

train_X, train_Y = preprocess(train_df)

# LSTM model
model = Sequential()
model.add(LSTM(32))
model.add(Dense(2, activation='softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

model.fit(train_X, train_Y, epochs=50, verbose=1)

test_X, test_Y = preprocess(test_df)

score, acc = model.evaluate(test_X, test_Y, verbose=2)
print('Test score:', score)
print('Test accuracy:', acc)