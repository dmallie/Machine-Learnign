# -*- coding: utf-8 -*-
"""
Created on Fri Dec  7 08:12:44 2018

@author: Dagmawi
"""
import numpy as np
from sklearn import linear_model, datasets, tree
import matplotlib.pyplot as plt

#Preparet the data
no_samples = 100
x = np.linspace(-np.pi, np.pi, no_samples)
y = 0.5*x + np.sin(x) + np.random.random(x.shape)


#split the dataset to training, validation & test
test_samples = np.random.permutation(no_samples)
#Training set
x_train = x[test_samples[:70]]
y_train = x[test_samples[:70]]
#validation set
x_validate = x[test_samples[70:85]]
y_validate = y[test_samples[70:85]]
#test ste
x_test = x[test_samples[85:]]
y_test = y[test_samples[85:]]
##
#draw a line that can fit to the data
#create an object that can model least squared error linear regression
LSE = linear_model.LinearRegression()
x_train_LSE = np.matrix(x_train.reshape(len(x_train),1))
y_train_LSE = np.matrix(y_train.reshape(len(y_train),1))
#Fit the line unto the training data
LSE.fit(x_train_LSE, y_train_LSE)
#plot the line
plt.scatter(x, y, color='red')
plt.plot(x.reshape((len(x),1)),LSE.predict(x.reshape((len(x),1))),color='blue')
plt.xlabel('x axis')
plt.ylabel('Values')
plt.title('Linear Regression')
plt.show()

# Evaluate the model, common approach is to use mean squared error on validation and test sets
mean_value_error=np.mean((y_validate - LSE.predict(x_validate.reshape(len(x_validate),1)))**2)
mean_test_error = np.mean((y_test - LSE.predict(x_test.reshape(len(x_test),1)))**2)
print ('Validation MSE: ',mean_value_error)
print ('Test MSE: ', mean_test_error)