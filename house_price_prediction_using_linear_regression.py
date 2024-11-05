##importing libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

##LOADING DATASET
dataset = pd.read_csv('dataset.csv')
# print(dataset.head())
# print(dataset.shape) #(1460, 2)

##Checking the data type of the dataset  
# dataset.info()

##Visualizing the data
# dataset.plot(kind='scatter', x='area', y='price')  
# plt.xlabel('Area')
# plt.ylabel('Price')
# plt.show()  



##Separating the data into X and y
X = dataset.drop('price', axis=1)
y = dataset.price
# print(X.head())
# print(y.head())

##model training
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X, y)

##model prediction
x=int(input("Enter the area(8k-20k) of the house you want to predict the price of: "))
# x=40000
X_pred = [[x]]
y_pred = model.predict(X_pred)
print(y_pred)

##Visualizing the model
# plt.scatter(X, y, color='blue', marker='o')
# plt.plot(X, y_pred, color='red')
# plt.xlabel('Area')
# plt.ylabel('Price')

##Theory Calculation
m=model.coef_[0]
b=model.intercept_
print("Theory Calculation",m,"x+",b)
print("The price of the house is:",m*x+b)

