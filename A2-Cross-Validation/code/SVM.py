# Ryan Russell
# July 13, 2020
# SENG 474 Assignment 2
# Directory: cd C:\Users\Ryan Russell\OneDrive - University of Victoria\Documents\Software Engineering\SENG 474\Assignments\A1\code

import numpy as np
from sklearn import metrics
from sklearn import svm
import matplotlib.pyplot as plt
import utils.mnist_reader as mnist_reader


# Reads, initializes, and reduces the size of the data from the fashion-MNIST dataset.
def init():

	x_train, y_train = mnist_reader.load_mnist('data/fashion', kind='train')
	x_test, y_test = mnist_reader.load_mnist('data/fashion', kind='t10k')

	x_train = np.array(x_train)
	y_train = np.array(y_train)
	x_test = np.array(x_test)
	y_test = np.array(y_test)

	x_train_temp = []
	y_train_temp = []
	x_test_temp = []
	y_test_temp = []
	sandal_count = 0
	sneaker_count = 0

	train_len = len(x_train)
	test_len = len(x_test)

	# Convert the training data into a binary classification problem.
	for i in range(train_len):

		# Let sandal data be class 0.
		if y_train[i] == 5 and sandal_count < 3000:
			y_train_temp.append(0)
			x_train_temp.append(x_train[i])
			sandal_count = sandal_count + 1

		# Let sneaker data be class 1.
		if y_train[i] == 7 and sneaker_count < 3000:
			y_train_temp.append(1)
			x_train_temp.append(x_train[i])
			sneaker_count = sneaker_count + 1

	# Convert the test data into a binary classification problem.
	for i in range(test_len):

		# Let sandal data be class 0.
		if y_test[i] == 5:
			y_test_temp.append(0)
			x_test_temp.append(x_test[i])

		# Let sneaker data be class 1.
		if y_test[i] == 7:
			y_test_temp.append(1)
			x_test_temp.append(x_test[i])

	x_train = np.array(x_train_temp) / 255
	y_train = y_train_temp
	x_test = np.array(x_test_temp) /255
	y_test = y_test_temp

	print("Training set size: " + str(len(x_train)))
	print("Test set size: " + str(len(x_test)))

	return x_train, y_train, x_test, y_test


# Implementation of Support Vector Machine using sklearn.
def SVM(x_train, y_train, x_test, y_test, c, gamma):

	#clf = svm.SVC(kernel = 'linear', C = c)
	clf = svm.SVC(kernel = 'rbf', C = c, gamma = gamma)
	clf.fit(x_train, y_train)

	y_test_pred = clf.predict(x_test)
	test_acc = metrics.accuracy_score(y_test, y_test_pred)
	train_acc = clf.score(x_train, y_train)

	print("C value " + str(c) + " complete.")

	return test_acc, train_acc


# This function is used to call SVM with variable C values and no k-fold validation.
def SVM_base():
	
	x_train, y_train, x_test, y_test = init()

	# 20 different C values will be used.
	
	# C values for midrange.
	# c_vals = [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000]

	# C values for underfitting.
	c_vals = [0.0000001, 0.0000005, 0.000001, 0.000005, 0.00001, 0.00005]

	# C values for overfitting.
	# c_vals = [10000, 50000, 100000, 500000, 1000000, 5000000]

	test_accs = []
	train_accs = []

	for c in c_vals:

		test_acc, train_acc = SVM(x_train, y_train, x_test, y_test, c, 1)
		test_accs.append(test_acc)
		train_accs.append(train_acc)

	create_base_graph(test_accs, train_accs, c_vals)


# Implementation of k-fold validation and optimally regularized Support Vector classifier.
def SVM_kfold():
	
	x_train, y_train, x_test, y_test = init()

	k_vals = [5, 6, 7, 8, 9, 10]
	c_vals = [0.085, 0.09, 0.095, 0.1, 0.105, 0.11, 0.115]

	test_accs = []

	for k in k_vals:

		print("Running k-fold for k = " + str(k) + ":")
		fold_accs = []
		x_length = len(x_train)
		y_length = len(y_train)
		fold = x_length / k

		for c in c_vals:

			current_c = []

			# Train the training set for each of the k unique groups.
			for group in range(k):

				next_group = group + 1
				start = int(fold * group)
				stop = int(fold * next_group)

				if group == (k - 1):

					x_train_training = x_train[0 : start]
					y_train_training = y_train[0 : start]

				elif group == 0:

					x_train_training = x_train[stop + 1 : x_length]
					y_train_training = y_train[stop + 1 : y_length]

				# The training set needs to be all the data in the group that is not used for testing.
				else:

					# Combine data on both sides of the k group.
					x_train_training_temp1 = np.array(x_train[0 : start])
					x_train_training_temp2 = np.array(x_train[stop + 1 : x_length])
					x_train_training = np.concatenate((x_train_training_temp1, x_train_training_temp2))

					# Combine data on both sides of the k group.
					y_train_training_temp1 = np.array(y_train[0 : start])
					y_train_training_temp2 = np.array(y_train[stop + 1 : y_length])
					y_train_training = np.concatenate((y_train_training_temp1, y_train_training_temp2))

				x_train_test = x_train[start : stop]
				y_train_test = y_train[start : stop]

				test_acc, train_acc = SVM(x_train_training, y_train_training, x_train_test, y_train_test, c, 1)
				current_c.append(test_acc)

			# Compute the average accuracy and compile the into a single list for plotting the graph.
			avg = sum(current_c) / len(current_c)
			fold_accs.append(avg)
			print()

		test_accs.append(fold_accs)

	create_kfold_graph(test_accs, c_vals)


# Implementation of k-fold validation and optimally regularized Gaussian Support Vector classifier.
def SVM_gaussian():
	
	x_train, y_train, x_test, y_test = init()

	# Selected k = 10 due to it having the highest accuracy for previous k-fold experiments.
	k = 10
	
	#c_vals = [0.05, 0.075, 0.1, 0.125, 0.15]
	#c_vals = [0.05]
	#run_name = "0.05"
	# c_vals = [0.075]
	# run_name = "0.075"
	#c_vals = [0.1]
	#run_name = "0.1"
	# c_vals = [0.125]
	# run_name = "0.125"
	c_vals = [0.15]
	run_name = "0.15"

	# Gamma values for all C.
	#gamma_vals = [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1]

	#Gamma values for single C.
	gamma_vals = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1]

	test_accs = []
	train_accs = []

	x_length = len(x_train)
	y_length = len(y_train)
	fold = x_length / k

	for c in c_vals:

		c_test_accs = []
		c_train_accs = []

		for gamma in gamma_vals:

			print("Running k-fold gaussian for gamma = " + str(gamma) + " and C = " + str(c) + ":")
			gamma_test_accs = []
			gamma_train_accs = []

			# Train the training set for each of the k unique groups.
			for group in range(k):

				next_group = group + 1
				start = int(fold * group)
				stop = int(fold * next_group)
				
				if group == (k - 1):

					x_train_training = x_train[0 : start]
					y_train_training = y_train[0 : start]

				elif group == 0:

					x_train_training = x_train[stop + 1 : x_length]
					y_train_training = y_train[stop + 1 : y_length]

				# The training set needs to be all the data in the group that is not used for testing.
				else:

					# Combine data on both sides of the k group.
					x_train_training_temp1 = np.array(x_train[0 : start])
					x_train_training_temp2 = np.array(x_train[stop + 1 : x_length])
					x_train_training = np.concatenate((x_train_training_temp1, x_train_training_temp2))

					# Combine data on both sides of the k group.
					y_train_training_temp1 = np.array(y_train[0 : start])
					y_train_training_temp2 = np.array(y_train[stop + 1 : y_length])
					y_train_training = np.concatenate((y_train_training_temp1, y_train_training_temp2))

				x_train_test = x_train[start : stop]
				y_train_test = y_train[start : stop]

				test_acc, train_acc = SVM(x_train_training, y_train_training, x_train_test, y_train_test, c, gamma)
				gamma_test_accs.append(test_acc)
				gamma_train_accs.append(train_acc)

			# Compute the average accuracy and compile the into a single list for plotting the graph.
			test_avg = sum(gamma_test_accs) / len(gamma_test_accs)
			c_test_accs.append(test_avg)
			train_avg = sum(gamma_train_accs) / len(gamma_train_accs)
			c_train_accs.append(train_avg)
			print()

		test_accs.append(c_test_accs)
		train_accs.append(c_train_accs)

	#create_gaussian_graph_allC(test_accs, gamma_vals)
	create_gaussian_graph(test_accs, train_accs, gamma_vals, run_name)


# Function to plot graph for the base Support Vector Machine model.
def create_base_graph(test_accs, train_accs, c_vals):
	
	plt.title("Support Vector Machine (Linear Kernel)")
	plt.xlabel("Regularization Parameter C")
	plt.xscale('log')
	plt.ylabel("Accuracy")

	plt.plot(c_vals, test_accs, 'g', label = "Test Accuracy")
	plt.plot(c_vals, train_accs, 'y', label = "Training Accuracy")

	plt.legend()

	filename = "SVM_underfit.png"
	plt.savefig(filename)
	plt.clf()


# Function to plot graph for the Support Vector Machine model with k-fold validation.
def create_kfold_graph(test_accs, c_vals):
	
	plt.title("Support Vector Machine (with K-Fold Validation and Linear Kernel)")
	plt.xlabel("Regularization Parameter C")
	plt.ylabel("Accuracy")

	plt.plot(c_vals, test_accs[0], 'r', label = "k = 5")
	plt.plot(c_vals, test_accs[1], 'g', label = "k = 6")
	plt.plot(c_vals, test_accs[2], 'b', label = "k = 7")
	plt.plot(c_vals, test_accs[3], 'y', label = "k = 8")
	plt.plot(c_vals, test_accs[4], 'k', label = "k = 9")
	plt.plot(c_vals, test_accs[5], 'm', label = "k = 10")

	plt.legend()

	filename = "SVM_kfold.png"
	plt.savefig(filename)
	plt.clf()	


# Function to plot graph for the Gaussian Support Vector Machine model with k-fold validation.
def create_gaussian_graph_allC(test_accs, gamma_vals):
	
	plt.title("Support Vector Machine (with K-Fold and Gaussian Kernel)")
	plt.xlabel("Gamma")
	plt.xscale('log')
	plt.ylabel("Accuracy")

	plt.plot(gamma_vals, test_accs[0], 'r', label = "C = 0.05")
	plt.plot(gamma_vals, test_accs[1], 'g', label = "C = 0.075")
	plt.plot(gamma_vals, test_accs[2], 'b', label = "C = 0.1")
	plt.plot(gamma_vals, test_accs[3], 'y', label = "C = 0.125")
	plt.plot(gamma_vals, test_accs[4], 'k', label = "C = 0.15")

	plt.legend()

	filename = "SVM_gaussian_allC.png"
	plt.savefig(filename)
	plt.clf()	


# Function to plot graph for the Gaussian Support Vector Machine model with k-fold validation.
def create_gaussian_graph(test_accs, train_accs, gamma_vals, run_name):
	
	title = "Support Vector Machine (with K-Fold and Gaussian Kernel) For C = " + run_name
	plt.title(title)
	plt.xlabel("Gamma")
	plt.xscale('log')
	plt.ylabel("Accuracy")

	plt.plot(gamma_vals, test_accs[0], 'r', label = "Test Accuracy")
	plt.plot(gamma_vals, train_accs[0], 'b', label = "Training Accuracy")

	plt.legend()

	filename = "SVM_gaussian_" + run_name + ".png"
	plt.savefig(filename)
	plt.clf()


def main():
	#SVM_base()
	#SVM_kfold()
	SVM_gaussian()


main()