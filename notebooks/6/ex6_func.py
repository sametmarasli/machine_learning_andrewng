import numpy as np
from sklearn import svm
import matplotlib.pyplot as plt

def plotData(X, y):
	'''
	PLOTDATA(x,y) plots the data points with + for the positive examples
	and o for the negative examples. X is assumed to be a Mx2 matrix.
	'''

	# seperate the data where y=1 and y=0
	positive = X[np.where(y==1)[0]]
	negative = X[np.where(y==0)[0]]

	# define bounds of the graph
	x_min, x_max = X[:, 0].min()-(X[:, 0].max()*0.1), X[:, 0].max()+(X[:, 0].max()*0.1)
	y_min, y_max = X[:, 1].min()-(X[:, 1].max()*0.1), X[:, 1].max()+(X[:, 1].max()*0.1)

	fig, ax = plt.subplots(figsize=(7,5))
	ax.scatter(positive[:,0], positive[:,1], c='k', s=60, marker='+', linewidths = 1, label='Positive')
	ax.scatter(negative[:,0], negative[:,1], c='y', s=60, linewidths = .5, label='Positive')
	ax.set_xlim(x_min, x_max)
	ax.set_ylim(y_min, y_max)
	ax.legend()


def SVM_classifiers(X,y,C,svc): 
	'''
	SVM_classifiers(X,scv)
	plots the decision surface for the SVM classifiers with linear kernels.
	'''

	# create a mesh to plot in
	x_min, x_max = X[:, 0].min()-(X[:, 0].max()*0.15), X[:, 0].max()+(X[:, 0].max()*0.15)
	y_min, y_max = X[:, 1].min()-(X[:, 1].max()*0.15), X[:, 1].max()+(X[:, 1].max()*0.15)
	xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01),
	                     np.arange(y_min, y_max, 0.01))

	# predict the points
	Z = svc.predict(np.c_[xx.ravel(), yy.ravel()])
	# Put the result into a color plot
	Z = Z.reshape(xx.shape)
	
	fig, ax = plt.subplots(1,2,figsize=(14,5))
	ax[0].contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.6)
	ax[0].scatter(X[:, 0], X[:, 1], s=40, c=y, cmap=plt.cm.coolwarm)
	ax[0].set_xlabel('X1')
	ax[0].set_ylabel('X2')
	ax[0].set_title('SVM (C={}) Decision Boundary and Area'.format(C))
	ax[0].set_xlim(xx.min(), xx.max())
	ax[0].set_ylim(yy.min(), yy.max())

	# Predict confidence scores for samples
	conf = svc.decision_function(X)

	ax[1].scatter(X[:,0], X[:,1], s=40, c=conf, cmap=plt.cm.coolwarm)
	ax[1].set_xlabel('X1')
	ax[1].set_ylabel('X2')
	ax[1].set_title('SVM (C={}) Decision Confidence'.format(C))
	ax[1].set_xlim(xx.min(), xx.max())
	ax[1].set_ylim(yy.min(), yy.max())



def visualizeBoundaryLinear(X,y,C,svc): 
	'''
	VISUALIZEBOUNDARYLINEAR(X,y,C,svc) plots a linear decision boundary 
	learned by the SVM and overlays the data on it
	'''

	# initialize the coefficents and intercept values
	coef = np.ravel(svc.coef_)
	inter = svc.intercept_

	# initialize some X axis values 
	x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
	xp = np.arange(x_min, x_max, 0.1)

	# calculate the y axis values
	yp = -(coef[0]*xp +  inter)/coef[1]

	# determine the positive and negative values
	positive = X[np.where(y==1)[0]]
	negative= X[np.where(y==0)[0]]

	fig, ax = plt.subplots(figsize=(7,5))

	ax.plot(xp,yp,'--b')
	# ax.scatter(X[:, 0], X[:, 1], s=40, c=y, cmap=plt.cm.coolwarm)
	ax.plot(positive[:,0], positive[:,1], 'k+', mew=2, ms=10, label='Positive')
	ax.plot(negative[:,0], negative[:,1], 'yo', mew=1, ms=10,label='Negative')
	ax.set_title('SVM (C={}) Decision Boundary'.format(C))
	ax.set_xlabel('X1')
	ax.set_ylabel('X2')
	ax.legend()


def gaussianKernel(x1, x2, sigma):
    x1 = x1.ravel()
    x2 = x2.ravel()
    return np.exp(-(x1-x2).dot((x1-x2))/(2.*np.square(sigma)))
