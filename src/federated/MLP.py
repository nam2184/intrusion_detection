# Multilayer Perceptron MLP
"""A multilayer perceptron is a logistic regressor where instead of feeding
the input to the logistic regression you insert a intermediate layer,
called the hidden layer, that has a nonlinear activation function
(usually tanh or sigmoid). One can use many such hidden layers making
the architecture deep
"""
from __future__ import print_function
import numpy as np
import tensorflow as tf
import math
import timeit
import os
from logisticRegression2 import LogisticRegression

class HiddenLayer(object):
	def __init__(self, input, n_inp, n_out, W = None, b = None, activation = tf.nn.sigmoid):
		"""
		Typical hidden layer of a MLP: units are fully-conected and have
		sigmoidal activation function. Weight matrix W is of shape (n_inp, n_out)
		and the bias vector b is of shape (n_out).
		Note : The nonlinear used here is sigmoid

		Hidden unit activation is given by sigmoi(matmul(input, W) + b)
		:param input: a sysbolic tensor of shape(n_examples, n_inp)
		:param n_inp: int, dimensionality of input
		:param n_out: int, number of hidden units
		:param activation: None linearity to be applied in the hidden layer
		"""

		self.input = input

		""" W is initialized with W_values which is uniformely sampled from
		4 * sqrt(-6/(n_inp + n_out)) and 4 * sqrt (6/(n_inp + n_out)) for
		sigmoid activation function
		"""
		if W is None:
			bound = 4.0 * math.sqrt(6.0 / (n_inp + n_out))
			W = tf.compat.v1.Variable(tf.compat.v1.random.uniform([n_inp, n_out], minval = -bound,
				maxval = bound), dtype = tf.float32, name = "W")

		if b is None:
			b = tf.compat.v1.Variable(tf.zeros([n_out]), dtype = tf.float32, name = "b")

		self.W = W
		self.b = b

		mean = tf.compat.v1.matmul(input, self.W) + self.b
		self.output = activation(mean) if activation is not None else mean

		# Params
		self.params = [self.W, self.b]
  
class MLP(object):
	""" Multi-Layer Perceptron class
	A multilayer perceptron is a feedforward artificial neural network model
	that has one layer or more of hidden units and nonlinear activations.
	Intermediate layers usually have as activation function tanh or the sigmoid
	function(defined here by a HiddenLayer class) while the top layer is
	a softmax layer(defined here by a LogisticRegression class)
	"""

	def __init__(self, input, n_inp, n_hidden, n_out):
		""" Initialize the parameters for the multilayer perceptron
		:param input: symbolic variable that describes the input of the architecture
		(one minibatch)
		:param n_inp: int, number of input units, the dimension of the space in
		which the datapoints lie
		:param n_hidden: int, number of hidden units
		:param n_out: int, number of output units, the dimension of the space in
		which the labels lie
		"""
		"""Since we are dealing with a one hidden layer MLP, this will traslate
		into a HiddenLayer with a sigmoid function connected to the LogisticRegression
		layer; the activation function can be replaced by tanh or any other
		nonlinear function
		"""

		self.hiddenLayer = HiddenLayer(input, n_inp = n_inp, n_out = n_hidden,
			activation = tf.nn.sigmoid)

		#The logistic regression layer gets as input the hidden units
		#of the hidden layer
		self.logRegressionLayer = LogisticRegression(self.hiddenLayer.output, n_inp = n_hidden, n_out = n_out)

		#L1 norm; one regularization option is to enforce L1 norm to be small
		self.L1 = tf.compat.v1.reduce_sum(tf.abs(self.hiddenLayer.W)) + tf.compat.v1.reduce_sum(tf.abs(self.logRegressionLayer.W))

		#square of L2 norm; one regularization option is to enforce square of L2 norm to be small
		self.L2_sqr = tf.compat.v1.reduce_sum(self.hiddenLayer.W **2) + tf.compat.v1.reduce_sum(self.logRegressionLayer.W **2)

		#cross entropy cost
		self.cost = self.logRegressionLayer.cost
		self.accuracy = self.logRegressionLayer.accuracy

		#the parameters of the model are the parameters of the two layer it is made out of
		self.params = self.hiddenLayer.params + self.logRegressionLayer.params

		self.input = input

