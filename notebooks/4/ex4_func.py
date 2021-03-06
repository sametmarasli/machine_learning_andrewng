import numpy as np


def sigmoid(z):
	'''
	J = SIGMOID(z) computes the sigmoid of z.
	'''
	return 1/(1+np.exp(-z))


def sigmoidGradient(z):
	'''
	g = SIGMOIDGRADIENT(z) computes the gradient of the sigmoid function
	evaluated at z. This should work regardless if z is a matrix or a
	vector. In particular, if z is a vector or matrix, you should return
	the gradient for each element.
	'''
	return sigmoid(z)*(1-sigmoid(z))


def randInitializeWeights(L_in, L_out):
	'''
	W = RANDINITIALIZEWEIGHTS(L_in, L_out) randomly initializes the weights 
	of a layer with L_in incoming connections and L_out outgoing 
	connections. 
	Note that W should be set to a matrix of size(L_out, 1 + L_in) as
	the column row of W handles the "bias" terms
	'''
	epsilon_init = np.sqrt(6)/np.sqrt(L_in + L_out)

	return (np.random.random([L_out, L_in+1])*2*epsilon_init-epsilon_init)



def nnCostFunction(nn_params, input_layer_size, hidden_layer_size, num_labels, X, y, lambda_=0.):
	'''
	[J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
	X, y, lambda) computes the cost and gradient of the neural network. The
	parameters for the neural network are "unrolled" into the vector
	nn_params and need to be converted back into the weight matrices. 
	The returned parameter grad should be a "unrolled" vector of the
	partial derivatives of the neural network.
	'''
	lambda_ = float(lambda_)

	# unroll parameters
	theta1 = np.reshape(nn_params[:hidden_layer_size*(input_layer_size+1)],(hidden_layer_size,input_layer_size+1))
	theta2 = np.reshape(nn_params[hidden_layer_size*(input_layer_size+1):],(num_labels,hidden_layer_size+1))

	m,n = X.shape

	theta1_grad = np.zeros(theta1.shape)
	theta2_grad = np.zeros(theta2.shape)
	J = 0

	for i in range(m):
	    a1 = np.insert(X[i,:], 0, 1,axis=0) #(401x1)
	    z2 = theta1.dot(a1)                 #(25x1)
	    a2 = sigmoid(z2)
	    a2 = np.insert(a2, 0, 1,axis=0)     #(26x1)
	    z3 = theta2.dot(a2)                 #(10x1)
	    a3 = sigmoid(z3)
	    fir =  y[i,:] * np.log(a3)
	    sec =  (1-y[i,:])*  np.log(1-a3)
	    J = J+ 1./m * np.sum(-(fir+sec))
	    
	    delta3 = a3-y[i,:]
	    
	    delta2 = (theta2[:,1:].T.dot(delta3) ) * sigmoidGradient(z2) #(25,)

	    theta1_grad = theta1_grad + 1./m * delta2[...,np.newaxis].dot(a1[...,np.newaxis].T)
	    
	    theta2_grad = theta2_grad + 1./m * delta3[...,np.newaxis].dot(a2[...,np.newaxis].T)


	# Cost fuction's regularization term
	thetas = np.concatenate([np.ravel(theta1[:,:input_layer_size]),
							 np.ravel(theta2[:,:hidden_layer_size])])
	reg = lambda_/(2*m) * thetas.dot(thetas)

	J = J + reg # add the regularization term

	
	# Gradients regularization terms
	theta1_grad[:,1:] = theta1_grad[:,1:]+lambda_/m*theta1[:,1:]

	theta2_grad[:,1:] = theta2_grad[:,1:]+lambda_/m*theta2[:,1:]


	grad = np.concatenate([np.ravel(theta1_grad),np.ravel(theta2_grad)])

	return J,grad



def debugInitializeWeights(fan_out, fan_in):
	'''
	W = DEBUGINITIALIZEWEIGHTS(fan_in, fan_out) initializes the weights 
	of a layer with fan_in incoming connections and fan_out outgoing 
	connections using a fix set of values
	Note that W should be set to a matrix of size(1 + fan_in, fan_out) as
	the first row of W handles the "bias" terms
	'''
	#Set W to zeros
	W_size = fan_out* (1 + fan_in)
	#Initialize W using "sin", this ensures that W is always of the same
	#values and will be useful for debugging
	W = np.array(np.sin(range(1,W_size+1))/10).reshape(fan_out,1 + fan_in)
	return W



def computeNumericalGradient(J, theta):
    '''
    numgrad = COMPUTENUMERICALGRADIENT(J, theta) computes the numerical
    gradient of the function J around theta. Calling y = J(theta) should
    return the function value at theta.
	
	Notes: The following code implements numerical gradient checking, and 
	returns the numerical gradient.It sets numgrad(i) to (a numerical 
	approximation of) the partial derivative of J with respect to the 
	i-th input argument, evaluated at theta. (i.e., numgrad(i) should 
	be the (approximately) the partial derivative of J with respect 
	to theta(i).)
    '''
    numgrad = np.zeros(theta.size);
    perturb = np.zeros(theta.size);
    e = 1e-4;
    for i in range(theta.size):
        perturb[i] = e
        loss1 =J(theta - perturb)[0]
        loss2 =J(theta + perturb)[0]
        numgrad[i] = (loss2 - loss1) / (2*e)
        perturb[i] = 0
    return numgrad


def predict(theta1, theta2, X):
	'''
	p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
	trained weights of a neural network (Theta1, Theta2)
	'''
	a1 = np.insert(X, 0, 1,axis=1)
	z2 = a1.dot(theta1.T)
	a2 = sigmoid(z2)
	a2 = np.insert(a2, 0, 1,axis=1)
	a2.shape, theta2.shape
	z3 = a2.dot(theta2.T)
	a3 = sigmoid(z3)
	y_hat = np.argmax(a3, axis=1)
	return y_hat

