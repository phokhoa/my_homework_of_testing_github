#Standard libraries
import random
#Third-party libraries
import numpy as np
import theano
import theano.tensor as T
from theano.tensor.nnet import softmax
from theano import function

def vectorized_result(j, numclass):
	e = np.zeros((numclass, 1))
	e[j] = 1.0
	return e

def ReLU(z): return T.maximum(0.0, z)


#### Main Network class
class FCNetwork():
	def __init__(self, sizes):
		self.num_layers = len(sizes)
		self.sizes = sizes
		self.weights = [theano.shared(np.asarray(np.random.normal(loc=0.0,scale=np.sqrt(1.0/y),size=(y, x)),dtype=theano.config.floatX),
						name='w',
						borrow=True)
						for x, y in zip(self.sizes[:-1], self.sizes[1:])]
		self.biases = [theano.shared(np.asarray(np.random.normal(loc=0.0,scale=2.0,size=(1,y)),dtype = theano.config.floatX),
						name='b',
						broadcastable = (True,False),
						borrow=True)
						for y in self.sizes[1:]]
		self.params = []
		for weight,bias in zip (self.weights,self.biases):
			self.params.append(weight)
			self.params.append(bias)

		x = T.matrix('x')
		y = T.matrix('y')
		y_subclass = T.matrix('y_subclass')
		a = x
		for i in xrange(len(self.weights)-2):
			z = T.dot(a,self.weights[i].transpose())+self.biases[i]
			a = ReLU(z)

		z = T.dot(a,self.weights[-2].transpose())+self.biases[-2]
		#a = ReLU(z)
		a = softmax(a)

		z = T.dot(a,self.weights[-1].transpose())+self.biases[-1]
		a_unified = softmax(z)



		epsilon = 10e-16
		lmbda = T.scalar('lmbda')
		eta = T.scalar('eta')
		#epsilon = 0.0
		self.cost = T.mean(-T.log(a[T.arange(a.shape[0]),T.argmax(y_subclass,axis=1)] + epsilon))
		for w in self.weights[:-1]:
			self.cost += 0.5*(lmbda)*T.mean(w**2 + epsilon)



		self.grads = [T.grad(self.cost,param) for param in self.params[:-4]]

		self.updates = [(param,param-eta*grad) for param,grad in zip(self.params[:-4],self.grads)]

		self.train = theano.function([x,y_subclass,y,eta,lmbda],self.cost,updates = self.updates,allow_input_downcast=True,on_unused_input='ignore')

		self.cost_unified = T.mean(-T.log(a_unified[T.arange(a_unified.shape[0]),T.argmax(y,axis=1)] + epsilon))
		for w in self.weights:
			self.cost_unified += 0.5*(lmbda)*T.mean(w**2 + epsilon)
		self.grads_unified = [T.grad(self.cost_unified,param) for param in self.params]
		self.updates_unified = [(param,param-eta*grad) for param,grad in zip(self.params,self.grads_unified)]
		self.train_unified = theano.function([x,y_subclass,y,eta,lmbda],self.cost_unified,updates = self.updates_unified,allow_input_downcast=True,on_unused_input='ignore')

		self.feedforward = theano.function([x],a, allow_input_downcast=True)
		self.feedforward_unified = theano.function([x],a_unified, allow_input_downcast=True)




	def SGD(self, training_data, epochs, mini_batch_size, eta, lmbda = 1.0):
		for j in xrange(epochs):
			n = len(training_data)
			random.shuffle(training_data)
			mini_batches = [training_data[k:k+mini_batch_size] for k in xrange(0, n, mini_batch_size)]
			for mini_batch in mini_batches:
				X = [np.array(a[0]) for a in mini_batch]
				Y_sub = [np.array(a[1]) for a in mini_batch]
				Y = [np.array(a[2]) for a in mini_batch]
				cost = self.train(X,Y_sub,Y,eta,lmbda/n)
				cost_unified = self.train_unified(X,Y_sub,Y,eta,lmbda/n)
			#print "Cost on training data: {}".format(cost)
	def feedforward_func(x):
		return self.feedforward(x)
