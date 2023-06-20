import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
import numpy as np

with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

test_df = pd.read_csv("Data/test_clean.csv")

test_reviews = test_df["clean_reviews"]

id = test_df['id']

max_length = 31

test_sequences = tokenizer.texts_to_sequences(test_reviews)
test_padded = pad_sequences(test_sequences, maxlen=max_length, padding="post", truncating="post")

model = tf.keras.models.load_model("tweets_classifier.h5")

predict = model.predict(test_padded).flatten()

predict_final = np.where(predict > 0.5, 1, 0)

submission = list(zip(id, predict_final))

submission_data = pd.DataFrame(submission, columns=['id', 'target'])

submission_data.to_csv("Data/my_submission.csv", index=False)