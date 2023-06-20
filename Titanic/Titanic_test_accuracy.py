import pandas as pd

data1 = pd.read_csv("Titanic data/gender_submission.csv")

data2 = pd.read_csv("Titanic data/my_submission.csv")

right = 0
wrong = 0

for i in range(data1.shape[0]):
    if data1["Survived"][i] == data2["Survived"][i]:
        right += 1
    else:
        wrong += 1

accuracy = right/(right+wrong)*100
print(accuracy)