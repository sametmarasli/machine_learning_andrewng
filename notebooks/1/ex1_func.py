import numpy as np


def featureNorm(X):   
    '''
    FEATURENORMALIZE(X) returns a normalized version of X where
    the mean value of each feature is 0 and the standard deviation
    is 1. This is often a good preprocessing step to do when
    working with learning algorithms.
    '''

    sigma = np.std(X,axis=0,ddof=1) # ddof=0 population st dev
    mu = np.mean(X,axis=0)
    X = (X-mu)/sigma
    return X,sigma,mu


def computeCost(X, y, theta):
    '''
    J = COMPUTECOST(X, y, theta) computes the cost of using theta as the
    parameter for linear regression to fit the data points in X and y
    '''

    m = len(y)
    residual = (X.dot(theta))-y
    J = np.dot(residual.T,residual)/(2*m)

    return J


def gradientDescent(X, y, theta, alpha, num_iters):
    '''
    theta = GRADIENTDESENT(X, y, theta, alpha, num_iters) updates theta by 
    taking num_iters gradient steps with learning rate alpha
    '''

    m = len(y)
    J_history = np.zeros(num_iters)

    for iter in range(num_iters):
        theta = theta - (alpha/m) * np.dot(X.T,(X.dot(theta)-y))
        J_history[iter] = computeCost(X, y, theta)

    return theta, J_history


def computeCostMulti(X, y, theta):
    '''    
    J = COMPUTECOSTMULTI(X, y, theta) computes the cost of using theta as the
    parameter for linear regression to fit the data points in X and y

    '''    

    m,n = X.shape
    theta = theta.reshape(n,1)
    residual = X.dot(theta)-y
    J = 1./(2*m) * residual.T.dot(residual)

    return J


def gradientDescentMulti(X, y, theta, alpha, num_iters):
    '''    
    GRADIENTDESCENTMULTI Performs gradient descent to learn theta
    theta = GRADIENTDESCENTMULTI(x, y, theta, alpha, num_iters) updates theta by
    taking num_iters gradient steps with learning rate alpha

     '''   

    m,n = X.shape
    theta = theta.reshape(n,1)
    J_val = np.zeros((num_iters,1))
    
    for i in range(num_iters):
        residual = np.dot(X,theta)-y
        grad = np.dot(X.T,residual)
        theta = theta - (alpha/m)*grad
        J_val[i] = computeCostMulti(X,y,theta)
    
    return theta,J_val

def normalEqn(X, y):    
    '''
    theta = pinv(X'X)X'y
    '''

    theta = np.dot(np.linalg.pinv(X.T.dot(X)),X.T.dot(y)) 

    return theta