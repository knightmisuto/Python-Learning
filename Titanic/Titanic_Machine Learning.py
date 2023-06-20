import numpy as np
import matplotlib.pyplot as plt
from Functions import *
import pandas as pd

#path đến file train
path = "Titanic data/train.txt"

#lấy file train
train_data = np.loadtxt(path, delimiter=" ")

#xây dựng x,y train
X = np.zeros([train_data.shape[0],train_data.shape[1]])
X[:,0] = 1
X[:,1:] = train_data[:,1:]
y = train_data[:,0]

#normalize data
[X,mu,s] = Normalize(X)

#train data
[Theta, J_hist] = GradientDescent(X,y,0.1,400)

#load file test data
test_data = np.loadtxt("Titanic data/test.txt")

inputs = np.zeros([test_data.shape[0],test_data.shape[1]+1])

inputs[:,0] = 1
inputs[:,1:] = test_data[:,:]

predicts = []

for value in inputs:
    value = (value-mu)/s
    value[0] = 1
    predict = 0
    for i in range(train_data.shape[1]):
        predict += (value[i]*Theta[i])
    predicts.append(int(round(predict,0)))

Submiss_data = pd.read_csv("Titanic data/test.csv")

Submiss_data = Submiss_data.drop(columns=["Name","Ticket","Embarked","Fare","Cabin","SibSp","Pclass","Sex","Age","Parch"])

Submiss_data["Survived"] = predicts

Submiss_data.to_csv("Titanic data/submission.csv", index=False)