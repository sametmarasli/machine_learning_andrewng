import numpy as np


def sigmoid(z):
    '''
    J = SIGMOID(z) computes the sigmoid of z.
    '''

    return 1./(1+np.exp(-z))


def costFunction(theta,X,y):
    '''
    J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
    parameter for logistic regression 
    '''
    
    m,n = X.shape
    h = sigmoid(np.dot(X,theta))
    J = (-np.dot(y.T,np.log(h))-np.dot((1-y).T,np.log(1-h)))*1/m
    
    return J


def gradient(theta,X,y):
    '''
    grad = gradient(theta, X, y)
    computes the gradient of the cost w.r.t. to the parameters.
    '''

    m,n = X.shape
    y = np.matrix(y)
    X = np.matrix(X)
    theta = np.matrix(theta)
    h = sigmoid(X.dot(theta.T))
    grad = (X.T*(h-y))/m

    return grad

def accuracy(X,y,theta):

    m,n = X.shape
    theta = theta.reshape(n,1)
    prob = sigmoid( X.dot(theta))

    return np.mean(np.array( prob>0.5, dtype='int_') == y)*100


def costFunctionReg(theta, X, y, lambda_):
    '''
    J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
    theta as the parameter for regularized logistic regression and the
    gradient of the cost w.r.t. to the parameters. 
    '''

    m,n = X.shape
    X = np.array(X)
    y = np.array(y)
    h = sigmoid(X.dot(theta))
    reg = lambda_/(2.*m)*(theta.T.dot(theta))
    J = -(y.T.dot(np.log(h))+(1-y).T.dot(np.log(1.-h)))*1./m + reg
  
    return J

def gradientReg(theta, X, y, lambda_):
    '''
    grad = gradient(theta, X, y, lambda) computes the 
    gradient of the cost w.r.t. to the parameters. 
    '''

    m,n = X.shape
    y = np.matrix(y)
    X = np.matrix(X)
    theta = np.matrix(theta)
    h = sigmoid(X.dot(theta.T))
    grad = X.T*(h-y)
    
    z = np.zeros(n) 
    z[1:]=1 
    reg = lambda_*(np.multiply(theta,z).T)
    
    grad = (grad + reg)/m

    return grad

def accuracyReg(theta,X,y):

    theta = np.matrix(theta)
    X = np.matrix(X)
    predict = (sigmoid(X*(theta.T))>=0.5).astype(int)
    
    return np.asscalar(sum(predict == y)/float(len(y)))*100