# -*- coding: utf-8 -*-
"""
Created on Fri Dec  7 10:14:42 2018

@author: Dagmawi
"""
import numpy as np
from sklearn import linear_model, datasets, tree
import matplotlib.pyplot as plt

#preparig the dataset
iris = datasets.load_iris()
X = iris.data[:,:2]
Y = iris.target
X = X[:100]
Y = Y[:100]
no_samples = len(Y)

# splitting the data into training, validation and test sets
random_indices = np.random.permutation(no_samples)
#Training set
noTrainingSamples = int(no_samples*0.7)
X_train = X[random_indices[:noTrainingSamples]]
Y_train = Y[random_indices[:noTrainingSamples]]
#Validation set
noValidationSamples = int(no_samples*0.15)
X_Validation = X[random_indices[noTrainingSamples:noTrainingSamples + noValidationSamples]]
Y_Validation = Y[random_indices[noTrainingSamples:noTrainingSamples + noValidationSamples]]
#Test set
noTestingSamples = int(no_samples*0.15)
X_test = X[random_indices[no_samples - noTestingSamples:no_samples]]
Y_test = Y[random_indices[no_samples - noTestingSamples:no_samples]]

# Visualizing the training data
X_class0 = np.asmatrix([X_train[i] for i in range(len(X_train)) if Y_train[i]==0]) #Picking only the first two classes
Y_class0 = np.zeros((X_class0.shape[0]),dtype=np.int)
X_class1 = np.asmatrix([X_train[i] for i in range(len(X_train)) if Y_train[i]==1])
Y_class1 = np.ones((X_class1.shape[0]),dtype=np.int)

plt.scatter(X_class0[:,0], X_class0[:,1],color='red')
plt.scatter(X_class1[:,0], X_class1[:,1],color='blue')
plt.xlabel('sepal length')
plt.ylabel('sepal width')
plt.legend(['class 0','class 1'])
plt.title('Fig 3: Visualization of training data')
plt.show()