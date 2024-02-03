# Logistic Regression
import numpy as np 
import tensorflow as tf 
import timeit
import os
#from tensorflow.examples.tutorials.mnist import input_data

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
		self.W = tf.Variable(tf.zeros([n_inp, n_out]))
		# Initialize the biases b as a vector of n_out 0s
		self.b = tf.Variable(tf.zeros([n_out]))

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
		return tf.reduce_mean( - tf.reduce_sum(y * tf.log(self.pred), reduction_indices = 1))

	def accuracy(self, y):
		""" Calculate accuracy
		:param y: desired output
		"""
		correct_prediction = tf.equal(self.correct, tf.argmax (y, axis = 1))
		return tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

if __name__ == "__main__":
	# Load mnist dataset
	mnist = input_data.read_data_sets("/tmp/data/", one_hot = True)

	# Parameters
	learning_rate = 0.01
	training_epochs = 500
	batch_size = 100
	display_step = 50
	n_inp =784
	n_out = 10

	# Define the input and target
	x = tf.placeholder(tf.float32, [None, n_inp])
	y = tf.placeholder(tf.float32, [None, n_out])
	logistic_regression = LogisticRegression(input = x, n_inp = n_inp, n_out = n_out)
	#Cost 
	cost = logistic_regression.cost(y)
	# Accuracy
	accuracy = logistic_regression.accuracy(y)
	# Gradient descent
	optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

	# Initialize the variables
	init = tf.global_variables_initializer()

	#Training
	with tf.Session() as sess:
		sess.run(init)

		#Traing cycle
		for epoch in range(training_epochs):
			avg_cost = 0.0
			total_batch = int(mnist.train.num_examples / batch_size)

			# Loop over all batchs
			for i in range(total_batch):
				batch_xs, batch_ys = mnist.train.next_batch(batch_size)
				#Run optimization op (backprop) and cost op (to get the loss value)
				_, c = sess.run([optimizer, cost], feed_dict = {x : batch_xs, y : batch_ys})
				# Compute average cost
				avg_cost += c/total_batch
			# Display logs per epoch step
			if epoch % display_step == 0:
				print("Epoch:", '%04d' % (epoch +1), "cost:", "{:.9f}".format(avg_cost))
				acc = sess.run(accuracy, feed_dict = {x: mnist.test.images, y: mnist.test.labels})
				print("Accuracy:", acc)
		print("Finished")
