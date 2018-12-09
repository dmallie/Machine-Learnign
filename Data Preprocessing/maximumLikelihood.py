import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#importing the dataset
dataset = pd.read_csv('RandomHeight2.csv')
X = dataset.iloc[:,0].values
Xs = np.msort(X)
#We calculate the mu and sigma
mu = np.mean(X)
sigma = np.std(X)
Y = [len(X)]
for i in range(len(X)-1):
    Y.append(0)
print Xs

#Calculate the normal distribution aka Gaussian distribution
#Y = 1/D*e^-(x-mu)^2/F
#where D = 1/sqrt(2*pi*sigma^2) and F = 2*sigma^2
D = 1/np.sqrt(2*np.pi*(sigma**2))
F = 2*(sigma**2)
for i in range(len(X)):
    Y[i] = (D)*(np.exp(-((Xs[i]-mu)**2)/F))
plt.plot(Xs, Y,linewidth=1, color='y')
plt.show()
