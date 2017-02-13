import cPickle
import numpy as np
from cifar import load_cifar
from cifar import load_cifar_file
import matplotlib.pyplot as plt
from logreg import sigmoid
# Try to do classification based on logistic regression for some CIFAR-10 images.
#IT will be difficult to get convergence using all 3072 features values and 10000-images in bach-1
# We will the images to grayscale to reduce the nof. features

trainimages, trainlab = load_cifar_file('data_batch_1')

#print trainlab.shape
#print trainimages.shape
#Reduce to gray level image by averaging the 3 bands


#Reshape images to vectors of one single column

# Remember to append a column of ones to the input data matrix.


#Use LogisticRegression from sklearn.linear_model

#If the algorithm does not converge, or takes too long, use only the first 500 images to train on.
# Display test accuracies (test on image 501:1000)  and training accuracies (image 1:500)


from sklearn.linear_model import LogisticRegression

#print training score

#print test score
# Why is the test score soo low?


