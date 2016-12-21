import numpy as np 
import matplotlib.pyplot as plt

def findClosestCentroids(X, centroids):
	'''
	idx = FINDCLOSESTCENTROIDS (X, centroids) returns the closest centroids
	in idx for a dataset X where each row is a single example. idx = m x 1 
	vector of centroid assignments (i.e. each entry in range [1..K])
	'''

	m,n = X.shape
	idx = np.zeros(m)

	for e in range(m):
	    idx[e] = np.argmin(np.sum((X[e,:] - 
	    	centroids)**2, axis=1))
	
	return idx


def computeCentroids(X, idx, K):
	'''
	centroids = COMPUTECENTROIDS(X, idx, K) returns the new centroids by 
	computing the means of the data points assigned to each centroid. It is
	given a dataset X where each row is a single data point, a vector
	idx of centroid assignments (i.e. each entry in range [1..K]) for each
	example, and K, the number of centroids. You should return a matrix
	centroids, where each row of centroids is the mean of the data points
	assigned to it.
	'''

	m,n = X.shape
	centroids = np.zeros((K,n))

	for  e in range(K):
	    c = np.where(idx == e)[0]
	    l =  c.size
	    centroids[e] =  1./l * np.sum( X[c], axis=0)

	return centroids



def plotkMeans(X, idx, centroids):
	'''
	PLOTPROGRESSKMEANS(X, idx, centroids) plots the data
	points with codelors assigned to each centroid. It also plots locations of the centroids.
	'''

	fig, ax = plt.subplots(figsize=(7,5))
	ax.scatter(X[:,0], X[:,1], c=idx, s=30,alpha=0.2)
	ax.scatter(centroids[:,0], centroids[:,1], c='m',marker='+', linewidths = 2, s=90)
	ax.set_xlabel('X1')
	ax.set_ylabel('X2')

    
def runkMeans(X, centroids, max_iters): 
	'''
	[centroids, idx] = RUNKMEANS(X, initial_centroids, max_iters,) 
	runs the K-Means algorithm on data matrix X, where each 
	row of X is a single example. It uses initial_centroids used as the
	initial centroids. max_iters specifies the total number of interactions 
	of K-Means to execute. runkMeans returns centroids, a Kxn matrix of the computed 
	centroids and idx, a m x 1 vector of centroid assignments (i.e. each entry in range [1..K])
	'''

	K,n = centroids.shape

	for i in range(max_iters):
		idx = findClosestCentroids(X, centroids)
		centroids = computeCentroids(X, idx, K)

	return centroids,idx

def kMeansInitCentroids(X, K):
	'''
	centroids = KMEANSINITCENTROIDS(X, K) returns K initial centroids to be
	used with the K-Means on the dataset X
	'''
	m,n = X.shape
	perm = np.random.permutation(m)
	initial_centroids = X[perm[:K],:]
	return initial_centroids

def featureNormalize(X):
	'''
	FEATURENORMALIZE(X) returns a normalized version of X where
	the mean value of each feature is 0 and the standard deviation
	is 1. This is often a good preprocessing step to do when
	working with learning algorithms.
	'''

	m,n = X.shape
	mu = np.mean(X, axis=0)
	sigma = np.std(X, axis=0, ddof=0)
	X_norm = (X-mu) / sigma
	return X_norm, mu, sigma


def pca(X):
	'''
	[U, S] = pca(X) computes eigenvectors of the covariance matrix of X
	Returns the eigenvectors U, the eigenvalues (on diagonal) in S
	'''

	m,n = X.shape
	sigma = 1./m*X.T.dot(X)
	U,S,V = np.linalg.svd(sigma)
	return U,S

def projectData(X, U, K):
	'''
	Z = projectData(X, U, K) computes the projection of 
	the normalized inputs X into the reduced dimensional space spanned by
	the first K columns of U. It returns the projected examples in Z.
	'''

	Z = X.dot(U[:,:K])
	return Z


def recoverData(Z, U, K):
	'''
	X_rec = RECOVERDATA(Z, U, K) recovers an approximation the 
	original data that has been reduced to K dimensions. It returns the
	approximate reconstruction in X_rec.
	'''

	X_rec = Z.dot(U[:,:K].T)
	return X_rec