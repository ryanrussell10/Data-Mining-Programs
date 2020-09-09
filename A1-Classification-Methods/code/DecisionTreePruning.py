# Ryan Russell
# June 21, 2020
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
from io import StringIO
from IPython.display import Image


# General function to plot graphs.
def create_graph(title, xlabel, ylabel, info, name):

	plt.title(title)
	plt.xlabel(xlabel)
	plt.ylabel(ylabel)

	plt.plot(info[0], 'r', label = "10.0%")
	plt.plot(info[1], 'g', label = "20.0%")
	plt.plot(info[2], 'b', label = "30.0%")
	plt.plot(info[3], 'y', label = "40.0%")
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


# Actual decision tree implementation.	
def tree(x_train, x_test, y_train, y_test, features, criterion, test_split, dataset):

	test_accuracies = []
	training_accuracies = []
	trees = []
	nodes = []

	clf = DecisionTreeClassifier(random_state = 0)
	path = clf.cost_complexity_pruning_path(x_train, y_train)
	alphas, impurities = path.ccp_alphas, path.impurities

	for alpha in alphas:

		clf = DecisionTreeClassifier(ccp_alpha = alpha, criterion = criterion, max_features = features)
		clf = clf.fit(x_train,y_train)
		y_pred = clf.predict(x_test)
		y_train_pred = clf.predict(x_train)

		if clf.tree_.node_count not in nodes:

			training_accuracies.append(metrics.accuracy_score(y_train, y_train_pred))
			test_accuracies.append(metrics.accuracy_score(y_test, y_pred))

			trees.append(clf)
			nodes.append(clf.tree_.node_count)

	# Below this line are the most accurate decision trees.

	# if dataset == "Cleveland" and criterion == 'gini' and features == 5 and test_split == 0.1:
	# 	plt.title("Cleveland Test Data and Training Data Accuracy")
	# 	plt.xlabel("# of Nodes")
	# 	plt.ylabel("Test Data Accuracy (Red) and Training Data Accuracy (Blue)")
	# 	plt.plot(nodes, test_accuracies, 'r')
	# 	plt.plot(nodes, training_accuracies, 'b')
	# 	plt.savefig('ClevelandAccuracies.png')

	# if dataset == "Banknote" and criterion == 'entropy' and features == 1 and test_split == 0.2:
	# 	plt.title("Banknote Test Data and Training Data Accuracy")
	# 	plt.xlabel("# of Nodes")
	# 	plt.ylabel("Test (Green) Accuracy and Training (Yellow) Accuracy")
	# 	plt.plot(nodes, test_accuracies, 'g')
	# 	plt.plot(nodes, training_accuracies, 'y')
	# 	plt.savefig('BanknoteAccuracies.png')

	max_value = max(test_accuracies)
	return max_value


def main():

	cleveland_gini_trees = []
	cleveland_entropy_trees = []
	banknote_gini_trees = []
	banknote_entropy_trees = []
	cleveland_max_features = 13
	banknote_max_features = 4

	criteria = ['gini', 'entropy']
	test_splits = [0.1, 0.2, 0.3, 0.4]

	for test_split in test_splits:

		x_train_cleveland, x_test_cleveland, y_train_cleveland, y_test_cleveland, num_cols_cleveland = cleveland_data(test_split)
		x_train_banknote, x_test_banknote, y_train_banknote, y_test_banknote, num_cols_banknote = banknote_data(test_split)

		for criterion in criteria:

			cleveland_trees = []
			banknote_trees = []

			for num_features in range(1, cleveland_max_features):
				cleveland_trees.append(tree(x_train_cleveland, x_test_cleveland, y_train_cleveland, y_test_cleveland, num_features, criterion, test_split, "Cleveland"))
			
			for num_features in range(1, banknote_max_features):
				banknote_trees.append(tree(x_train_banknote, x_test_banknote, y_train_banknote, y_test_banknote, num_features, criterion, test_split, "Banknote"))

			if criterion == 'entropy':
				cleveland_entropy_trees.append(cleveland_trees)
				banknote_entropy_trees.append(banknote_trees)

			elif criterion == 'gini':
				cleveland_gini_trees.append(cleveland_trees)
				banknote_gini_trees.append(banknote_trees)

	create_graph("Cleveland Data (Split Criterion = Entropy)", "# of Features", "Accuracy", cleveland_entropy_trees, "EntropyCleveland")
	create_graph("Cleveland Data (Split Criterion = Gini)", "# of Features", "Accuracy", cleveland_gini_trees, "GiniCleveland")
	create_graph("Banknote Data (Split Criterion = Entropy)", "# of Features", "Accuracy", banknote_entropy_trees, "EntropyBanknote")
	create_graph("Banknote Data (Split Criterion = Gini)", "# of Features", "Accuracy", banknote_gini_trees, "GiniBanknote")


main()