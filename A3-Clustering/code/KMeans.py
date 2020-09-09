# Ryan Russell
# July 26, 2020
# SENG 474 Assignment 3
# Directory: cd C:\Users\Ryan Russell\OneDrive - University of Victoria\Documents\Software Engineering\SENG 474\Assignments\A3\code

import numpy as np
import matplotlib.pyplot as pyplot
import pandas as pd
import random as rand
from mpl_toolkits.mplot3d import Axes3D


def kmeans2D():

	# Read in the 2D data
	df = pd.read_csv('dataset1.csv')

	iterations = 10
	x_val = df.iloc[:, [0, 1]].values
	m = x_val.shape[0]

	arr = np.array([])

	for K in range(2, 21, 2):
		
		out = {}
		centroids = np.array([]).reshape(x_val.shape[1], 0)

		# Uniform Random Initialization
		# for k in range(K):
		# 	random = rand.randint(0, m - 1)
		# 	centroids = np.c_[centroids, x_val[random]]

		# k-means++ Initialization
		centroids = kmeansplusplus(x_val, K)

		# Lloyd's Algorithm fitting
		for i in range(iterations):

			dists = np.array([]).reshape(m, 0)

			for k in range(K):
				result = (x_val - centroids[:, k])**2
				dist = np.sum(result, axis = 1)
				dists = np.c_[dists, dist]

			# Argmin calculation
			cluster = np.argmin(dists, axis = 1) + 1

			for k in range(K):
				out[k + 1] = np.array([]).reshape(2, 0)

			for n in range(m):
				out[cluster[n]] = np.c_[out[cluster[n]], x_val[n]]

			# Transpose each entry in the output array
			for k in range(K):
				out[k + 1] = out[k + 1].T

			for k in range(K):
				centroids[:, k] = np.mean(out[k + 1], axis = 0)

		centroids = centroids.T
		sse = 0

		# Retrieve sum of squared errors and append each to an array for output
		for k in range(K):
			result2 = (out[k + 1] - centroids[k, :])**2
			sse += np.sum(result2)
		print(sse)
		arr = np.append(arr, sse)
		pyplot.title("K-means Algorithm on Dataset 1 with " + str(K) + " Clusters")

		# Plot the cluster map
		for k in range(K):
			pyplot.scatter(out[k + 1][:, 0], out[k + 1][:, 1])

		# Create a scatter plot for the centroids
		pyplot.scatter(centroids[:, 0], centroids[:, 1], s = 350, c = 'red', label = 'Centroids')
		#cur_file = "kmeans2D_" + str(K)
		cur_file = "kmeans++2D_" + str(K)
		pyplot.savefig(cur_file)
		pyplot.clf()

	#plotcost2D(arr, 'dataset1.csv')


def kmeans3D():
	
	# Read in the 3D data
	df = pd.read_csv('dataset2.csv')

	iterations = 10
	x_val = df.iloc[:, [0, 1, 2]].values
	m = x_val.shape[0]

	arr = np.array([])

	for K in range(2, 21, 2):
		
		out = {}
		centroids = np.array([]).reshape(x_val.shape[1], 0)

		# Uniform Random Initialization
		# for k in range(K):
		# 	random = rand.randint(0, m - 1)
		# 	centroids = np.c_[centroids, x_val[random]]

		# k-means++ Initialization
		centroids = kmeansplusplus(x_val, K)

		# Lloyd's Algorithm fitting
		for i in range(iterations):

			dists = np.array([]).reshape(m, 0)

			for k in range(K):
				result = (x_val - centroids[:, k])**2
				dist = np.sum(result, axis = 1)
				dists = np.c_[dists, dist]

			# Argmin calculation
			cluster = np.argmin(dists, axis = 1) + 1

			for k in range(K):
				out[k + 1] = np.array([]).reshape(3, 0)

			for n in range(m):
				out[cluster[n]] = np.c_[out[cluster[n]], x_val[n]]

			# Transpose each entry in the output array
			for k in range(K):
				out[k + 1] = out[k + 1].T

			for k in range(K):
				centroids[:, k] = np.mean(out[k + 1], axis = 0)

		centroids = centroids.T
		sse = 0

		# Retrieve sum of squared errors and append each to an array for output
		for k in range(K):
			result2 = (out[k + 1] - centroids[k, :])**2
			sse += np.sum(result2)
		print(sse)
		arr = np.append(arr, sse)

		fig = pyplot.figure()
		a = Axes3D(fig)
		a.set_title("K-means Algorithm on Dataset 2 with " + str(K) + " Clusters")

		# Plot the cluster map
		for k in range(K):
			a.scatter(out[k + 1][:, 0], out[k + 1][:, 1], out[k + 1][:, 2])

		# Create a scatter plot for the centroids
		a.scatter(centroids[:, 0], centroids[:, 1], centroids[:, 2], s = 350, c = 'red', label = 'Centroids')
		#cur_file = "kmeans3D_" + str(K)
		cur_file = "kmeans++3D_" + str(K)
		pyplot.savefig(cur_file)
		pyplot.clf()

	#plotcost(arr, 'dataset2.csv')


# Initialization method with better performance than uniform random initialization
def kmeansplusplus(x_val, K):
	
	random_val = rand.randint(0, x_val.shape[0])
	centroid = np.array([x_val[random_val]])

	# Loop through for each k value up to K
	for k in range(1, K):

		random_val = 0
		random_val2 = rand.random()
		setup_arr = np.array([])

		for cur_x in x_val:
			x_result = np.min(np.sum((cur_x - centroid)**2))
			setup_arr = np.append(setup_arr, x_result)

		# Calculate cumulative probability and init new random variable
		prob = setup_arr / np.sum(setup_arr)
		cumprob = np.cumsum(prob)

		for i, j in enumerate(cumprob):
			if random_val2 < j:
				random_val = i
				break

		centroid = np.append(centroid, [x_val[random_val]], axis = 0)
	
	return centroid.T


# Plots a graph of the cost vs. number of clusters
def plotcost(sse_arr, dataset):
	
	cluster_arr = np.arange(1, 21)
	pyplot.title('Cost vs. # of Clusters for ' + dataset)
	pyplot.plot(cluster_arr, sse_arr, color = 'green')
	pyplot.xlabel('# of Clusters')
	pyplot.ylabel('Cost (Sum of Squared Errors)')

	#filename = "costplot_2Dkmeans"
	#filename = "costplot_2Dkmeans++"
	#filename = "costplot_3Dkmeans"
	filename = "costplot_3Dkmeans++"
	pyplot.savefig(filename)
	pyplot.clf()


def main():
	kmeans2D()
	kmeans3D()


main()