import pandas as pd
import numpy as np
import tensorflow as tf
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

print(len(train_dataset.keys()))