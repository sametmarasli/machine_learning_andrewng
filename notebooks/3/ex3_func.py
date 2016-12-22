import numpy as np
import random as rd
import matplotlib.pyplot as plt
import matplotlib.cm as cm # used to display images in a specific colormap
from scipy.misc import toimage # used to convert matrix to image

def sigmoid(x):
    '''
    J = SIGMOID(z) computes the sigmoid of z.
    '''

    return 1/(1+np.exp(-x))


def lrCostFunction(theta,X,y,lambda_):
    '''
    J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
    theta as the parameter for regularized logistic regression and the
    gradient of the cost w.r.t. to the parameters. 
    '''

    m,n = X.shape
    h = sigmoid(X.dot(theta))
    reg = lambda_/(2.*m)*theta.T.dot(theta)
    J = -(y.T.dot(np.log(h))+(1-y).T.dot(np.log(1-h)))*1./m + reg

    return np.asscalar(J)


def gradient(theta,X,y,lambda_):
    '''
    grad = gradient(theta, X, y, lambda) computes the 
    gradient of the cost w.r.t. to the parameters. 
    '''

    m,n = X.shape
    theta = np.matrix(theta).T
    h = sigmoid(X.dot(theta))
    grad = X.T.dot(h-y)
    
    z = np.zeros((n,1))
    z[1:]=1
    reg = lambda_*np.multiply(theta,z)
    
    grad = (grad + reg)/m

    return grad


def predict(theta1,theta2,X):    
    '''
    #p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
    #trained weights of a neural network (Theta1, Theta2)
    '''    
    m,n = X.shape
    z2 = X.dot(theta1.T)
    a2 = sigmoid(z2)
    a2_b = np.insert(a2, 0, 1,axis=1)
    z2 = a2_b.dot(theta2.T)
    a3 = sigmoid(z2)
    pred = (np.argmax(a3,axis=1)+1).reshape(m,1)

    return pred

def permuter(X,y,pred):

	m,n = X.shape
	# chose a random data point
	sel = rd.sample(range(1, m), 1)
	# plot some images
	fig = plt.figure(figsize=(2,2))
	data1 = X[sel].reshape(20,20).T
	img = toimage(data1)
	plt.imshow(img,cmap = cm.Greys_r)
	print 'Displaying Example Image of {}'.format(float(y[sel]))
	print 'and the neural networks prediction is {}'.format(float(pred[sel]))

	return