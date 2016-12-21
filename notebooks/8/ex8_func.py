import numpy as np 
import matplotlib.pyplot as plt


def estimateGaussian(X):
	'''
	[mu sigma2] = estimateGaussian(X), 
	The input X is the dataset with each n-dimensional data point in one row
	The output is an n-dimensional vector mu, the mean of the data set
	and the variances sigma^2, an n x 1 vector
	'''

	mu = np.mean(X, axis=0)
	sigma2 = np.var(X, axis=0, ddof=0)

	return mu, sigma2


def estimateMulti(X):
	'''
	[mu_multi sigma2_multi] = estimateMulti(X)
	Computes the mean of the data set and the covariance matrix Sigma
	'''

	m,n = X.shape

	mu_multi = np.mean(X, axis=0)
	sigma2_multi = (X-mu_multi).T.dot(X-mu_multi)/float(m)

	return mu_multi,sigma2_multi


def multivariateGaussian(X, mu, sigma2):
	'''
	p = MULTIVARIATEGAUSSIAN(X, mu, Sigma2) Computes the probability 
	density function of the examples X under the multivariate gaussian 
	distribution with parameters mu and Sigma2. If Sigma2 is a matrix, it is
	treated as the covariance matrix. If Sigma2 is a vector, it is treated
	as the \sigma^2 values of the variances in each dimension (a diagonal
	covariance matrix)
	'''

	m,n = X.shape

	if sigma2.shape == (n,):
		sig_t = np.zeros((n,n))
		np.fill_diagonal(sig_t,sigma2)
		sigma2 = sig_t
		# print sigma2


	fir = 1/(np.power((2*np.pi),(n/2))*np.sqrt(np.linalg.det(sigma2)))
	# print fir
	sec = np.exp(-.5 * np.sum(np.multiply((X-mu).dot(np.linalg.pinv(sigma2)),(X-mu)),axis=1))

	p = fir*sec

	return p


def selectThreshold(yval, pval):
	'''
	[bestEpsilon bestF1] = SELECTTHRESHOLD(yval, pval) finds the best
	threshold to use for selecting outliers based on the results from a
	validation set (pval) and the ground truth (yval).
	'''

	stepsize = (max(pval)-min(pval)) / 1000
	best_F1 = 0
	best_epsilon = 0 

	for epsilon in np.arange(min(pval), max(pval), stepsize):

	    est = np.array((pval < epsilon).reshape(len(pval),1), dtype=int)

	    tpos = np.sum((yval == 1) & (est == 1), dtype=float)
	    fpos = np.sum((yval == 0) & (est == 1), dtype=float)
	    fneg = np.sum((yval == 1) & (est == 0), dtype=float)

	    prec = tpos / (tpos+fpos)
	    recall = tpos / (tpos+fneg)
	    
	    F1 = (2*prec*recall) / (prec+recall)

	    if F1 > best_F1:
	        best_F1 = F1
	        best_epsilon = epsilon
	        
		return best_F1, best_epsilon


def cofiCostFunc(params, Y, R, num_users, num_movies, num_features, lambda_ = 0.):
	'''
	[J, grad] = COFICOSTFUNC(params, Y, R, num_users, num_movies,
	num_features, lambda) returns the cost and gradient for the
	collaborative filtering problem.
	'''

	X = params[0:num_features*num_movies].reshape(num_movies, num_features)
	Theta = params[num_features*num_movies	:].reshape(num_users, num_features)
	
	error = (np.multiply(X.dot(Theta.T),R) - Y)
	J = 1./2 * np.sum(np.square(error))
	reg = lambda_/2*np.sum(np.square(Theta)) + lambda_/2*np.sum(np.square(X))
	J = J + reg

	X_grad = error.dot(Theta) + lambda_*X
	Theta_grad =  error.T.dot(X) + lambda_*Theta

	grad = np.concatenate((np.ravel(X_grad), np.ravel(Theta_grad)))
	
	return J,grad


def computeNumericalGradient(J, theta):
	'''
	numgrad = COMPUTENUMERICALGRADIENT(J, theta) computes the numerical
    gradient of the function J around theta. Calling y = J(theta) should
    return the function value at theta.
    '''

	numgrad = np.zeros(theta.size)
	perturb = np.zeros(theta.size)
	e = 1e-4;

	for i in range(theta.size):
		perturb[i] = e
		loss1 =J(theta - perturb)[0]
		loss2 =J(theta + perturb)[0]
		numgrad[i] = (loss2 - loss1) / (2*e)
		perturb[i] = 0

	return numgrad


def normalizeRatings(Y, R):
	'''
	[Ynorm, Ymean] = NORMALIZERATINGS(Y, R) normalized Y so that each movie
	has a rating of 0 on average, and returns the mean rating in Ymean.
	'''

	mu = np.sum(Y,axis=1,dtype=float) / np.sum(R,axis=1)
	mu = mu.reshape(mu.size,1)
	Ynorm = np.subtract(Y,mu)

	return np.multiply(Ynorm,R), mu
