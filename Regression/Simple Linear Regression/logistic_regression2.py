# -*- coding: utf-8 -*-
"""
Created on Fri Dec  7 08:12:44 2018

@author: Dagmawi
"""
import numpy as np
from sklearn import linear_model, datasets, tree
import matplotlib.pyplot as plt

# Data generation
noSamples = 100
lowerBound = -5
upperBound = 5
x = np.linspace(lowerBound, upperBound, noSamples)
y = 0.5*x + np.sin(x) + np.random.random(x.shape)

#splitting the dataset into training, validation & test
randomSamples = np.random.permutation(noSamples)
set1 = int (0.7*noSamples)
set2 = int (0.7*noSamples + 0.15*noSamples)
#Training set

x_train = x[randomSamples[:set1]]
y_train = x[randomSamples[:set1]]
#validation set
x_validate = x[randomSamples[set1:set2]]
y_validate = y[randomSamples[set1:set2]]
#test ste
x_test = x[randomSamples[set2:]]
y_test = y[randomSamples[set2:]]
##
#draw a line that can fit to the data
#create a decision tree regression object
treeDepth = np.arange(10)+1
trainErr = []
validationErr = []
testErr = []

for depth in treeDepth:
    model = tree.DecisionTreeRegressor(max_depth = depth)
    x_trainFit = np.matrix(x_train.reshape(len(x_train), 1))
    y_trainFit = np.matrix(y_train.reshape(len(y_train), 1))
    
    #fit the line to the training data
    model.fit(x_trainFit, y_trainFit)
    
    #visualize the data through graph
    plt.figure()
    plt.scatter(x_train, y_train, color='black')
    plt.plot(x.reshape(len(x),1), model.predict(x.reshape(len(x),1)), color='blue')
    plt.xlabel('X input')
    plt.ylabel('Y value')
    plt.title('Line fit to training data' + str(depth))
    plt.show()
    
    meanTrainErr = np.mean((y_train - model.predict(x_train.reshape(len(x_train), 1)))**2)
    meanValErr = np.mean((y_validate - model.predict(x_validate.reshape(len(x_train), 1)))**2)
    meanTestErr = np.mean((y_test - model.predict(x_test.reshape(len(x_test), 1)))**2)
    
    trainErr.append(meanTrainErr)
    validationErr.append(meanValErr)
    testErr.append(meanTestErr)
    
    print ('Training MSE: ', meanTrainErr, '\nValidation MSE: ', meanValErr, '\nTest MSE: ', meanTestErr)

plt.figure()
plt.plot(trainErr,c='red')
plt.plot(validationErr,c='blue')
plt.plot(testErr,c='green')
plt.legend(['Training error', 'Validation error', 'Test error'])
plt.title('Variation of error with maximum depth of tree')
plt.show()