import numpy as np
import scipy.optimize as opt
import matplotlib.pyplot as plt


def linearRegCostFunction(theta, X, y, lambda_ = 0.):
	'''
	[J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
	cost of using theta as the parameter for linear regression to fit the 
	data points in X and y.  Returns the cost in J and the gradient in grad
	'''
	m,n = X.shape
	theta = theta.reshape(n,1)
	cost = (X.dot(theta) - y).T.dot( (X.dot(theta)-y)) /(2.*m)
	reg = lambda_ / (2.*m) * theta[1:,:].T.dot(theta[1:,:])
	cpst = cost+reg

	grad = X.T.dot(X.dot(theta) - y)/m
	z = np.ones((n,1))
	z[0,:] = 0
	reg = np.multiply(lambda_/m * theta, z)
	grad = grad + reg

	return cost,grad


def trainLinearReg(X, y, lambda_):
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
	[error_train, error_val] = ...
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
	axes.plot(range(1,m+1),error_train)
	axes.plot(range(1,m+1),error_val)
	axes.set_ylim([0,error_val.max()])
	axes.set_xlim([0,m])
	axes.set_title('Learning curve for linear regression')
	axes.set_xlabel('Number of training examples')
	axes.set_ylabel('Error')

	# matlab style api alternative 
	# plt.figure(figsize=(7,4))
	# plt.axis([0, m, 0, error_val.max()])
	# plt.plot(range(1,m+1),error_train)
	# plt.plot(range(1,m+1),error_val)
	# plt.xlabel('Number of training examples')
	# plt.ylabel('Error')
	# plt.title('Learning curve for linear regression')

	return error_train, error_val


# k = 0.01
# p = [0]
# for e in range(11):
# 	p.append(k)
# 	k = k*2
# print p