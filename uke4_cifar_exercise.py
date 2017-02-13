import cPickle
import numpy as np
import matplotlib.pyplot as plt
# Try to do classification based on logistic regression for some CIFAR-10 images.
#IT will be difficult to get convergence using all 3072 features values and 10000-images in bach-1
# We will the images to grayscale to reduce the nof. features

def unpickle(file):
    import cPickle
    fo = open(file, 'rb')
    dict = cPickle.load(fo)
    fo.close()
    return dict

batch = unpickle('data_batch_1')
images = np.array(batch['data'])
lab = np.array(batch['labels'])

#print lab.shape
#print images.shape
#Reduce to gray level image by averaging the 3 bands


#Reshape images to vectors of one single column

# Remember to append a column of ones to the input data matrix.
X = np.ones((images.shape[0], images.shape[1]+1))
X[:, 1:] = images

#Use LogisticRegression from sklearn.linear_model

#If the algorithm does not converge, or takes too long, use only the first 500 images to train on.
# Display test accuracies (test on image 501:1000)  and training accuracies (image 1:500)
train_samples = 2000
train_X = X[:train_samples, :]
train_y = lab[:train_samples]
test_X = X[train_samples:2*train_samples, :]
test_y = lab[train_samples:2*train_samples]

from sklearn.linear_model import LogisticRegression
clf = LogisticRegression(n_jobs=4)
clf.fit(train_X, train_y)
print "Training: ", clf.score(train_X, train_y)
print "Test: ", clf.score(test_X, test_y)

#print training score

#print test score
# Why is the test score soo low?


