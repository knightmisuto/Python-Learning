import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import LabelEncoder

model = tf.keras.models.load_model('House_price_model.h5')

test_data = pd.read_csv("House price data/test.csv")

data_test_num = test_data.select_dtypes(exclude=['object'])
data_test_num.fillna(0, inplace=True)

data_test_cat = test_data.select_dtypes(include=['object'])
data_test_cat.fillna('NONE', inplace=True)

col_test_cat = list(data_test_cat.columns)

enc = LabelEncoder()

for col in col_test_cat:
    data_test_cat[col] = data_test_cat[col].astype('str')
    data_test_cat[col] = enc.fit_transform(data_test_cat[col])

data_test = data_test_num.merge(data_test_cat, left_index=True, right_index=True)
data_test = pd.get_dummies(data_test, prefix='', prefix_sep='')
data_test.pop("Id")

test_stats = data_test.describe()
test_stats = test_stats.transpose()

def norm(x):
  return (x - test_stats['mean']) / test_stats['std']

normed_test_data = norm(data_test)

predictions = model.predict(normed_test_data).flatten()

data = {"Id": test_data['Id'], 'SalePrice': predictions}
data = pd.DataFrame(data)

data.to_csv("House price data/my submission.csv", index=False)