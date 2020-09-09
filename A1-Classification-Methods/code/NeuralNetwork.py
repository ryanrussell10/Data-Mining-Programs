# Ryan Russell
# June 23, 2020
# SENG 474 Assignment 1
# Directory: cd C:\Users\Ryan Russell\OneDrive - University of Victoria\Documents\Software Engineering\SENG 474\Assignments\A1\Code
# Please see project README for references.

import math
import pandas as pd
import itertools
import pydotplus
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.tree import export_graphviz
from sklearn import tree as tree_util
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report,confusion_matrix
from io import StringIO
from IPython.display import Image


# General function to plot graphs.
def create_graph(title, xlabel, ylabel, info, name):

	plt.title(title)
	plt.xlabel(xlabel)
	plt.ylabel(ylabel)

	plt.plot(info[0], 'r', label = "0.001 Learning Rate")
	plt.plot(info[1], 'g', label = "0.01 Learning Rate")
	plt.plot(info[2], 'b', label = "0.1 Learning Rate")
	plt.plot(info[3], 'y', label = "0.2 Learning Rate")

	plt.legend()

	filename = name + ".png"
	plt.savefig(filename)
	plt.clf()


# Sets up the program for input from the Cleveland data set.
def cleveland_data(test_split):
	columns = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'num']
	feature_columns = columns[:13]
	cleveland_data = pd.read_csv("../cleaned_processed.cleveland.data", header = None, names = columns)
	cleveland_data.head()
	x = cleveland_data[feature_columns]
	y = cleveland_data.num # Target

	x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = test_split, random_state = 1)

	return x_train, x_test, y_train, y_test, 13	


# Sets up the program for input from the Banknote data set.
def banknote_data(test_split):

	columns = ['variance', 'skewness', 'curtosis', 'entropy', 'banknote_class']
	feature_columns = columns[:4]
	banknote_data = pd.read_csv("../data_banknote_authentication.csv", header = None, names = columns)
	banknote_data.head()
	x = banknote_data[feature_columns]
	y = banknote_data.banknote_class # Target

	x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = test_split, random_state = 1)

	return x_train, x_test, y_train, y_test, 4


# Actual neural network implementation.
def network(x_train, x_test, y_train, y_test, learning_rate, layer_sizes, iterations, dataset):
	
	clf = MLPClassifier(hidden_layer_sizes = (layer_sizes), max_iter = iterations, learning_rate_init = learning_rate)
	clf.fit(x_train, y_train)

	y_test_pred = clf.predict(x_test)
	y_train_pred = clf.predict(x_train)

	return metrics.accuracy_score(y_test, y_test_pred)

def main():
	
	cleveland_5layer_networks = []
	banknote_5layer_networks = []
	cleveland_10layer_networks = []
	banknote_10layer_networks = []
	cleveland_50layer_networks = []
	banknote_50layer_networks = []

	learning_rates = [0.001, 0.01, 0.1, 0.2]
	hidden_layer_sizes = [5, 10, 50]
	training_iterations = [1, 10, 20]

	for rate in learning_rates:

		x_train_cleveland, x_test_cleveland, y_train_cleveland, y_test_cleveland, num_cols_cleveland = cleveland_data(0.2)
		x_train_banknote, x_test_banknote, y_train_banknote, y_test_banknote, num_cols_banknote = banknote_data(0.2)

		for num_layers in hidden_layer_sizes:

			cleveland_networks = []
			banknote_networks = []

			for num_iterations in training_iterations:

				for iteration in range(1, num_iterations):

					cleveland_networks.append(network(x_train_cleveland, x_test_cleveland, y_train_cleveland, y_test_cleveland, rate, num_layers, iteration, "Cleveland"))
					banknote_networks.append(network(x_train_banknote, x_test_banknote, y_train_banknote, y_test_banknote, rate, num_layers, iteration, "Banknote"))

			if num_layers == 5:

				cleveland_5layer_networks.append(cleveland_networks)
				banknote_5layer_networks.append(banknote_networks)

			elif num_layers == 10:

				cleveland_10layer_networks.append(cleveland_networks)	
				banknote_10layer_networks.append(banknote_networks)

			elif num_layers == 50:

				cleveland_50layer_networks.append(cleveland_networks)
				banknote_50layer_networks.append(banknote_networks)

	create_graph("Cleveland Data (Hidden Layer Size = 5)", "# of Iterations", "Accuracy", cleveland_5layer_networks, "Cleveland5")
	create_graph("Banknote Data (Hidden Layer Size = 5)", "# of Iterations", "Accuracy", banknote_5layer_networks, "Banknote5")
	create_graph("Cleveland Data (Hidden Layer Size = 10)", "# of Iterations", "Accuracy", cleveland_10layer_networks, "Cleveland10")
	create_graph("Banknote Data (Hidden Layer Size = 10)", "# of Iterations", "Accuracy", banknote_10layer_networks, "Banknote10")
	create_graph("Cleveland Data (Hidden Layer Size = 50)", "# of Iterations", "Accuracy", cleveland_50layer_networks, "Cleveland50")
	create_graph("Banknote Data (Hidden Layer Size = 50)", "# of Iterations", "Accuracy", banknote_50layer_networks, "Banknote50")


main()