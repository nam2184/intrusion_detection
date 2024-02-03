import numpy as np
import tensorflow as tf
import timeit
import os

class LogisticRegression(object):
	"""Multi class Logistic Regression Class
	The logistic regression if fully described by a weight matrix W
	and bias vector b. Classification is done by projecting data points
	onto a set of hyperplanes, the distance to which is used to determine
	a class membership probability.
	"""

	def __init__(self, input, n_inp, n_out):
		"""Initialize the parameters of the logistic regression
		:param input: tensor, describes the input of the architecture (one minibatch)
		:param n_inp: int, number of input units, the dimension of the space
		in which the datapoints lie
		:param n_out: int, number of output units, the dimension of the space
		in which the labels lie
		"""

		# Initialize with 0 the weights W as a matrix of shape (n_inp, n_out)
		self.W = tf.compat.v1.Variable(tf.zeros([n_inp, n_out]))
		# Initialize the biases b as a vector of n_out 0s
		self.b = tf.compat.v1.Variable(tf.zeros([n_out]))

		# symbolic expression for computing the matrix of class-membership probabilities
		# Where:
		# W is a matrix where column-k represent the separation hyperplane for class-k
		# x is a matrix where row-j represents input training sample-j
		# b is a vector where element-k represent the free parameter of hyperplane-k

		self.pred = tf.nn.softmax(tf.matmul(input, self.W) + self.b)

		#symbolic description of how to compute prediction as class whose perbability
		# is maximal
		self.correct = tf.argmax(self.pred, axis = 1)
		# parameters of the model
		self.params = [self.W, self.b]

		#keep track of model input
		self.input = input


	def cost(self, y):
		"""Minimize the error using cross entropy
		:param y: tensor, corresponds to a vector that gives for example the correct label
		"""
		return tf.compat.v1.math.reduce_mean( - tf.compat.v1.reduce_sum(y * tf.math.log(self.pred), reduction_indices = 1))

	def accuracy(self, y):
		""" Calculate accuracy
		:param y: desired output
		"""
		correct_prediction = tf.equal(self.correct, tf.argmax (y, axis = 1))
		return tf.compat.v1.math.reduce_mean(tf.cast(correct_prediction, tf.float32))

