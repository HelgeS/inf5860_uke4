from __future__ import division

from sklearn import datasets


import matplotlib.pyplot as plt
import seaborn as sns

import pandas as pd
import numpy as np
import math


data = datasets.load_iris()
X1 = data.data[:100, :2]
y = data.target[:100]
X_full = data.data[:100, :]

setosa = plt.scatter(X1[:50,0], X1[:50,1], c='b')
versicolor = plt.scatter(X1[50:,0], X1[50:,1], c='r')
plt.xlabel("Sepal Length")
plt.ylabel("Sepal Width")
plt.legend((setosa, versicolor), ("Setosa", "Versicolor"))
plt.show()

# Remember to add a column of ones to X, as the first column
X = np.ones((100, 3))
X[:, 1:] = X1

def logistic_func(theta, x):
    #Fill in the correct value for the sigmoid function, the logistic function, not logarithm
    # If the input x is a vector, the output should be a vector
    sigmoidvalue = 1/(1+np.exp(-x.dot(theta)))
    return sigmoidvalue

def log_gradient(theta, x, y):
    #Compute the gradient of theta to use in gradient descent updates, without learning rate
    #All nfeat elements in theta should be
    theta_gradient = (logistic_func(theta, x) - np.squeeze(y)).T.dot(x)
    return theta_gradient


def cost_func(theta, x, y):
    # Compute the cost function for logistic regression
    sigmoids = logistic_func(theta, x)
    y = np.squeeze(y)
    costval = -y * np.log(sigmoids) - (1-y)*np.log(1-sigmoids)
    return np.mean(costval)


def grad_desc(theta_values, X, y, lr=.01, converge_change=.001):
    #Do gradient descent with learning rate lr and stop of the nof. changes is below limit
    #Return the resulting theta values, and an array with the cost values for each iteration
    # Stop if the abs(cost(it)-cost(it+1))<convergence_change
    cost = [(0, cost_func(theta_values, X, y))]
    cost_diff = 1
    it = 0

    while abs(cost_diff) > converge_change:
        it += 1
        theta_values = theta_values - (lr * log_gradient(theta_values, X, y))
        cur_cost = cost_func(theta_values, X, y)
        cost.append((it, cur_cost))
        cost_diff = cost[it][1] - cost[it-1][1]

    return theta_values, np.array(cost)


def pred_values(theta, X):
    "Predict the class labels"
    logval = logistic_func(theta, X)
    pred_value = np.round(logval)

    return pred_value

#X should be with an extra column added
shape = X.shape[1]
y_flip = np.logical_not(y) #flip Setosa to be 1 and Versicolor to zero to be consistent
betas = np.zeros(shape)
fitted_values, cost_iter = grad_desc(betas, X, y_flip)
print(fitted_values)
# Your theta-vector should be about 0.20 -1.22 2.07

predicted_y = pred_values(fitted_values, X)
predicted_y

correct = np.sum(y_flip == predicted_y)
print('correc', correct)

plt.plot(cost_iter[:,0], cost_iter[:,1])
plt.ylabel("Cost")
plt.xlabel("Iteration")
plt.show()

nofsamp,nfeat = X1.shape
def test_cost_func():
    Xtest = X[:10,:]
    #x0vec = np.zeros((5, nfeat))
    #X0append = np.ones((5, nfeat + 1))
    #X0append[:, 1:] = x0vec
    ytest = y[:10]

    thetatest = np.zeros(nfeat+1)
    thetatest[0]=0.0
    thetatest[1] = 1.0
    thetatest[2]= 1.0
    testcost = cost_func(thetatest, Xtest, ytest)
    correctcost = 8.17
    assert np.abs(testcost-correctcost)<0.5

def test_log_gradient():
    Xtest = X[:10, :]

    ytest = y[:10]
    thetatest = np.zeros(nfeat + 1)
    thetatest[0] = 0.0
    thetatest[1] = 0.0
    thetatest[2] = 0.0
    testthetagrad = log_gradient(thetatest, Xtest, ytest)
    correcttheta = np.zeros((nfeat+1))
    #print testthetagrad
    correcttheta[0] = 5.0
    correcttheta[1]=24.3
    correcttheta[2]=16.55
    diff = np.abs(testthetagrad-correcttheta)
    assert np.mean(diff)<0.1
