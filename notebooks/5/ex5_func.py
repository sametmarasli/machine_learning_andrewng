import numpy as np
import scipy.optimize as opt
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures



def linearRegCostFunction(theta, X, y, lambda_ = 0.):
	'''
	[J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
	cost of using theta as the parameter for linear regression to fit the 
	data points in X and y.  Returns the cost in J and the gradient in grad
	'''

	m,n = X.shape
	theta = theta.reshape(n,1)

	cost = 1 /(2.*m) * (X.dot(theta) - y).T.dot( (X.dot(theta)-y))
	reg = lambda_ / (2.*m) * theta[1:,:].T.dot(theta[1:,:])
	cost = cost+reg

	grad = 1./m * X.T.dot(X.dot(theta) - y)
	z = np.ones((n,1))
	z[0,:] = 0
	reg = np.multiply(lambda_/m * theta, z)
	grad = grad + reg

	return cost,grad


def trainLinearReg(X, y, lambda_ = 0.):
	'''
	[theta] = TRAINLINEARREG (X, y, lambda) trains linear regression using
	the dataset (X, y) and regularization parameter lambda. Returns the
	trained parameters theta.
	'''

	# initialize some values
	m,n = X.shape
	theta = np.zeros(n)

	# Create "short hand" for the cost function to be minimized
	costFunction = lambda p : linearRegCostFunction(p, X, y, lambda_)
	cost,grad  = costFunction(theta)

	# minimize the cost function
	fmin = opt.minimize(fun=costFunction, 
                    	x0=theta,
                    	method='TNC', 
                    	jac=True, 
                    	options={'maxiter':150})
	theta = fmin.x

	return theta


def	learningCurve(X, y, Xval, yval, lambda_):
	'''
	LEARNINGCURVE(X, y, Xval, yval, lambda) returns the train and
	cross validation set errors for a learning curve. In particular, 
	it returns two vectors of the same length - error_train and 
	error_val. Then, error_train(i) contains the training error for
	i examples (and similarly for error_val(i)).	
	'''
	
	m,n = X.shape

	error_train= np.zeros(m)
	error_val= np.zeros(m)

	for e in range(m):

		X_temp, y_temp = X[:e+1,:], y[:e+1,:] 

		theta = trainLinearReg(X_temp, y_temp, lambda_)
		error_train[e], grad = linearRegCostFunction(theta, X_temp, y_temp)
		error_val[e], grad = linearRegCostFunction(theta, Xval, yval)

	print 'Training Examples  Train Error  Cross Validation Error\n'
	for e in range(m):
	    print '\t{}\t \t{} \t\t{}'.format(e,round(error_train[e],2), round(error_val[e],2))

    #matplotlib's object oriented api
	fig1, axes = plt.subplots(figsize=(7,4))
	axes.plot(range(1,m+1),error_train,label="Train")
	axes.plot(range(1,m+1),error_val,label="Cross validation")
	axes.set_ylim([0,error_val.max()])
	axes.set_xlim([0,m])
	axes.set_title('Polynomial Regression Learning Curve (lambda = {})'.format(lambda_))
	axes.set_xlabel('Number of training examples')
	axes.set_ylabel('Error')
	axes.legend()

	# matlab style api alternative 
	# plt.figure(figsize=(7,4))
	# plt.axis([0, m, 0, error_val.max()])
	# plt.plot(range(1,m+1),error_train)
	# plt.plot(range(1,m+1),error_val)
	# plt.xlabel('Number of training examples')
	# plt.ylabel('Error')
	# plt.title('Learning curve for linear regression')

	#return error_train, error_val
	return


def featureNormalize(X, mu =np.zeros(1), sigma =np.zeros(1)):
	'''
	[X_norm_w_bias, mu, sigma] = FEATURENORMALIZE(X) returns a normalized version 
	of X where 	the mean value of each feature is 0 and the standard deviation
	is 1. This is often a good preprocessing step to do when working with learning algorithms.
	'''
	
	# drop the bias terms
	X_wb = X[:,1:]

	
	# if mu and sigma does not exist 
	if not(mu.any() and sigma.any()): 

		mu = np.mean(X_wb,axis=0)
		sigma = np.std(X_wb,axis=0,ddof=1) # degrees of freedom is 1 (sample st dev)
		X_norm = (X_wb-mu)/sigma
	
	# if mu and sigma assigned before
	else:	

		X_norm = (X_wb - mu) / sigma
		X_norm_w_bias = np.insert(X_norm,0,1,axis=1)
		return X_norm_w_bias
	
	# add bias terms back
	X_norm_w_bias = np.insert(X_norm,0,1,axis=1)

	return X_norm_w_bias, mu, sigma


def plotFit( X, y, lambda_, mu, sigma, theta, p_degree ):
	'''
	PLOTFIT( X, y, lambda_, mu, sigma, theta, p_degree 	) plots the learned polynomial
	fit with power p and feature normalization (mu, sigma).
	'''

	# initialize polynoial features
	poly =PolynomialFeatures(degree=p_degree,interaction_only=False,include_bias=True)

	# initialize values 
	vals = np.arange(np.min(X)-20, np.max(X)+20, 1)
	vals = vals.reshape(vals.size,1)

	# get polynomial terms for initialized vals
	vals_p = np.array(poly.fit_transform(vals))

	# normalize the initialized vals
	vals_p_n = featureNormalize(vals_p, mu, sigma)

	plt.figure(figsize=(7,4))
	plt.plot(vals,vals_p_n.dot(theta),'k--')
	plt.plot(X,y,'rx',mew=1,ms=10)
	plt.xlabel("Change in water level (x)")
	plt.ylabel('Water flowing out of the dam (y)')
	plt.title('Polynomial Regression Fit (lambda = {})'.format(lambda_))
	return 



def validationCurve(X, y, Xval, yval):
	'''
	[lambda_vec[pos]] = validationCurve(X, y, Xval, yval) returns the train
	and validation errors (in error_train, error_val) for different values of lambda. 
	You are given the training set (X, y) and validation set (Xval, yval).

	'''

	# initialize lambda values
	lambda_vec = [0, 0.01, 0.02, 0.04, 0.08, 0.16, 0.32, 0.64, 1.28, 2.56, 5.12, 10.24]
	l = len(lambda_vec)

	# initialize error vecs
	error_train = np.zeros(l)
	error_val = np.zeros(l)

	for i in range(l):

	    theta = trainLinearReg(X,y,lambda_vec[i])
	    # calc cost while lambda is 0
	    error_train[i],grad = linearRegCostFunction(theta, X ,y)
	    error_val[i],grad  = linearRegCostFunction(theta, Xval , yval) 

	# find the position of lambda gives the closest error 
	pos =  np.argmin(np.absolute(error_val - error_train))

	fig1,axes = plt.subplots(figsize = (7,4))
	axes.plot(lambda_vec, error_train,label="Train")
	axes.plot(lambda_vec, error_val,label="Cross validation")
	axes.plot(lambda_vec[pos], error_train[pos], 'rx', mew=2, ms=7)
	axes.set_ylim([0, error_val.max()])
	axes.set_xlim([0, max(lambda_vec)])
	axes.set_title('Learning curve for linear regression for diffrent lambdas')
	axes.set_xlabel('lambda')
	axes.set_ylabel('Error')
	axes.legend()
	return 	lambda_vec[pos]

def testError(X,y,X_test,ytest,lambda_):
	'''
	[error_test] = testError(X,y,X_test,ytest,lambda_) calculates the theta from train set
	and fit to the test set in order to estimate  and return the generalization error
	'''


	theta = trainLinearReg(X,y,lambda_)
	error_test , grad = linearRegCostFunction(theta, X_test, ytest)

	return error_test


def bootstrapLearningCurve(X, y, Xval, yval, lambda_, iteration):
	'''
	bootstrapLEARNINGCURVE(X, y, Xval, yval, lambda) returns the average results 
	across multiple sets of  randomly  selected  examples  to  determine  the  training
	error  and  cross validation error.
	'''

    # initialize some variables

	m,n = X.shape
	error_train= np.zeros((iteration,m))
	error_val= np.zeros((iteration,m))

	for i in range(iteration):                      

	    for e in range(m):
	        #  create permuted copies
	        perm = np.random.permutation(m)
	        X_temp, y_temp = X[perm,:][:e+1,:], y[perm,:][:e+1,:] 

	        theta = trainLinearReg(X_temp, y_temp, lambda_)
	        error_train[i,e], grad = linearRegCostFunction(theta, X_temp, y_temp)
	        error_val[i,e], grad = linearRegCostFunction(theta, Xval, yval)

	# the averaged errors        
	error_train =  np.mean(error_train,axis=0)
	error_val =  np.mean(error_val,axis=0)

	# #matplotlib's object oriented api
	fig1, axes = plt.subplots(figsize=(7,4))
	axes.plot(range(1,m+1),error_train,label="Train")
	axes.plot(range(1,m+1),error_val,label="Cross validation")
	axes.set_ylim([0,error_val.max()])
	axes.set_xlim([0,m])
	axes.set_title('Polynomial Regression Learning Curve (lambda = {})'.format(lambda_))
	axes.set_xlabel('Number of training examples')
	axes.set_ylabel('Error')
	axes.legend()