# -*- coding: utf-8 -*-
"""
Created on Wed Dec  5 07:56:25 2018

@author: Dagmawi
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LinearRegression
#import the dataset

data = pd.read_excel('Salary.xlsx')
X = data.iloc[:,:-1].values
Y = data.iloc[:,1].values

#Splitting the dataset into the Training set and Test set
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 1/5, random_state = 0)

#Fitting linear regression into training set
regressor = LinearRegression()
regressor.fit(X_train, Y_train)
 
#Predicting based on test set
Y_prediction = regressor.predict(X_train)

#plotting result

plt.scatter(X_train, Y_train, color='red')
plt.plot(X_train, Y_prediction, color='blue')
plt.title('Linear regression')
plt.xlabel('Experience in years')
plt.ylabel('Salary')
plt.show()
