import pandas as pd
import numpy as np

data = pd.read_csv("Titanic data/test.csv")

#loại bỏ những cột không cần thiết
data = data.drop(columns=["Name","Ticket","Embarked","Fare","Cabin","PassengerId","SibSp","Parch"])
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

np.savetxt(r'Titanic data/test.txt', data.values, fmt="%.1f")