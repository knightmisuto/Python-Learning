import pandas as pd
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier
import matplotlib.pyplot as plt

data = pd.read_csv("Titanic data/train.csv")

#loại bỏ những cột không cần thiết
data = data.drop(columns=["Name","Ticket","Embarked","Fare","Cabin","PassengerId"])
#thay những giá trị Nan = 0
data = data.replace(np.nan,0)

#hàm thay đổi giới tính sang 1 0. Nam: 0, Nữ: 1
def Sex_to_number(sex):
    if sex == 'male':
        return 0
    if sex == 'female':
        return 1

#áp dụng hàm vào thay đổi giá trị
data['Sex'] = data['Sex'].apply(Sex_to_number)

#lấy các dữ liệu của features X
X = data.iloc[:,1:]
#lấy dữ liệu output y
y = data.iloc[:,0]

#model phân loại rẽ nhánh
model = ExtraTreesClassifier()
model.fit(X,y)

#sắp xếp và hiển thị những features được dùng nhiều nhất trong data
feat_importances = pd.Series(model.feature_importances_, index=X.columns)
feat_importances.nlargest(10).plot(kind='barh')
plt.show()