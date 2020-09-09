# Ryan Russell
# July 28, 2020
# SENG 474 Assignment 3
# Directory: cd C:\Users\Ryan Russell\OneDrive - University of Victoria\Documents\Software Engineering\SENG474\Assignments\A3\code

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
from scipy.cluster.hierarchy import dendrogram
from sklearn.cluster import AgglomerativeClustering as agg


def single_link_cluster():
	
	#data = 'dataset1.csv'
	data = 'dataset2.csv'

	# Read in the data
	df = pd.read_csv(data)

	#x_val = df.iloc[:, [0, 1]].values
	x_val = df.iloc[:, [0, 1, 2]].values

	hac = agg(n_clusters = None, distance_threshold = 1, linkage = 'single')
	hac.fit(x_val)

	# Formulate matrix of linkages for dendrogram plotting
	linkages = create_dendrogram(hac)
	dendrogram(linkages, truncate_mode = 'lastp')

	plt.title("Dendrogram for Single Linkage HAC on " + data)
	#plt.savefig("singleHAC2D_dendrogram.png")
	plt.savefig("singleHAC3D_dendrogram.png")
	plt.clf()

	#single_cluster2D(x_val)
	single_cluster3D(x_val)


def average_link_cluster():
	
	#data = 'dataset1.csv'
	data = 'dataset2.csv'

	# Read in the data
	df = pd.read_csv(data)

	#x_val = df.iloc[:, [0, 1]].values
	x_val = df.iloc[:, [0, 1, 2]].values

	hac = agg(n_clusters = None, distance_threshold = 0, linkage = 'average')
	hac.fit(x_val)

	# Formulate matrix of linkages for dendrogram plotting
	linkages = create_dendrogram(hac)
	dendrogram(linkages, truncate_mode = 'lastp')

	plt.title("Dendrogram for Average Linkage HAC on " + data)
	#plt.savefig("averageHAC2D_dendrogram.png")
	plt.savefig("averageHAC3D_dendrogram.png")
	plt.clf()

	#average_cluster2D(x_val)
	#average_cluster3D(x_val)


# Formulates the dendrogram and creates the figure for output
def create_dendrogram(hac):

	n = len(hac.labels_)
	sample_counts = np.zeros(hac.children_.shape[0])

	for i, j in enumerate(hac.children_):

		count = 0
		for child in j:
			if child < n:
				count = count + 1
			else:
				count += sample_counts[child - n]

		sample_counts[i] = count

	# Format linkages as a matrix
	linkages = np.column_stack([hac.children_, hac.distances_, sample_counts])

	# Change type of data to float and return the matrix
	linkages = linkages.astype(float)
	return linkages


# Creates a scatter plot for the 2D data using single linkage
def single_cluster2D(x_val):
	
	hac = agg(n_clusters = 28, affinity = 'euclidean', linkage = 'single')
	hac.fit_predict(x_val)

	plt.title("Cluster Map for Single Linkage HAC on dataset1.csv")
	plt.scatter(x_val[:, 0], x_val[:, 1], c = hac.labels_)
	plt.savefig("singleHAC2D_cluster.png")
	plt.clf()


# Creates a scatter plot for the 2D data using average linkage
def average_cluster2D(x_val):
	
	hac = agg(n_clusters = 3, affinity = 'euclidean', linkage = 'average')
	hac.fit_predict(x_val)

	plt.title("Cluster Map for Average Linkage HAC on dataset1.csv")
	plt.scatter(x_val[:, 0], x_val[:, 1], c = hac.labels_)
	plt.savefig("averageHAC2D_cluster.png")
	plt.clf()


# Creates a scatter plot for the 3D data using single linkage
def single_cluster3D(x_val):
	
	hac = agg(n_clusters = 28, affinity = 'euclidean', linkage = 'single')
	hac.fit_predict(x_val)

	fig = plt.figure()

	a = Axes3D(fig)
	a.set_title("Cluster Map for Single Linkage HAC on dataset2.csv")
	a.scatter(x_val[:, 0], x_val[:, 1], x_val[:, 2], c = hac.labels_)

	plt.savefig("singleHAC3D_cluster.png")


# Creates a scatter plot for the 3D data using average linkage
def average_cluster3D(x_val):
	
	hac = agg(n_clusters = 26, affinity = 'euclidean', linkage = 'average')
	hac.fit_predict(x_val)

	fig = plt.figure()

	a = Axes3D(fig)
	a.set_title("Cluster Map for Single Linkage HAC on dataset2.csv")
	a.scatter(x_val[:, 0], x_val[:, 1], x_val[:, 2], c = hac.labels_)

	plt.savefig("averageHAC3D_cluster.png")


def main():
	single_link_cluster()
	#average_link_cluster()


main()
