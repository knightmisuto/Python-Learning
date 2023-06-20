import pandas as pd
import string
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

data_train = pd.read_csv("Data/train.csv", index_col="id")
data_test = pd.read_csv("Data/test.csv", index_col="id")

data_train = data_train.drop_duplicates(subset=['text','target'],
                                        keep='first')

pstem = PorterStemmer()

def clean_text(text):
    text = text.lower()
    text = re.sub('[0-9]', '', text)
    text = "".join([char for char in text if char not in string.punctuation])
    tokens = word_tokenize(text)
    tokens = [pstem.stem(word) for word in tokens]
    text = ' '.join(tokens)
    return text

data_train["clean_reviews"] = data_train["text"].apply(clean_text)
data_test["clean_reviews"] = data_test["text"].apply(clean_text)

train_clean = data_train[["clean_reviews", "target"]]
test_clean = data_test["clean_reviews"]

train_clean.to_csv("Data/train_clean.csv")
test_clean.to_csv("Data/test_clean.csv")