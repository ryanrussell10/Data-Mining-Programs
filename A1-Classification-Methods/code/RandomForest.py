# Ryan Russell
# June 22, 2020
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

	plt.plot(info[0], 'r', label = "1")
	plt.plot(info[1], 'g', label = "5")
	plt.plot(info[2], 'b', label = "10")
	plt.plot(info[3], 'y', label = "50")
	plt.plot(info[4], 'k', label = "100")
	plt.plot(info[5], 'm', label = "500")

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


# Actual random forest implementation.	
def forest(x_train, x_test, y_train, y_test, features, criterion, num_trees, dataset):

	clf = RandomForestClassifier(criterion = criterion, n_estimators = num_trees, max_features = features)
	clf.fit(x_train, y_train)

	y_test_pred = clf.predict(x_test)
	y_train_pred = clf.predict(x_train)

	return metrics.accuracy_score(y_train, y_train_pred)


def main():
	
	cleveland_gini_forests = []
	cleveland_entropy_forests = []
	banknote_gini_forests = []
	banknote_entropy_forests = []
	cleveland_max_features = 13
	banknote_max_features = 4

	criteria = ['gini', 'entropy']
	num_trees = [1, 5, 10, 50, 100, 500]

	for tree_amount in num_trees:

		x_train_cleveland, x_test_cleveland, y_train_cleveland, y_test_cleveland, num_cols_cleveland = cleveland_data(0.2)
		x_train_banknote, x_test_banknote, y_train_banknote, y_test_banknote, num_cols_banknote = banknote_data(0.2)

		for criterion in criteria:

			cleveland_forests = []
			banknote_forests = []

			for num_features in range(1, cleveland_max_features):
				cleveland_forests.append(forest(x_train_cleveland, x_test_cleveland, y_train_cleveland, y_test_cleveland, num_features, criterion, tree_amount, "Cleveland"))
				
			for num_features in range(1, banknote_max_features):
				banknote_forests.append(forest(x_train_banknote, x_test_banknote, y_train_banknote, y_test_banknote, num_features, criterion, tree_amount, "Banknote"))

			if criterion == 'entropy':
				cleveland_entropy_forests.append(cleveland_forests)
				banknote_entropy_forests.append(banknote_forests)

			elif criterion == 'gini':
				cleveland_gini_forests.append(cleveland_forests)
				banknote_gini_forests.append(banknote_forests)

	create_graph("Cleveland Data (Split Criterion = Entropy)", "# of Features", "Accuracy", cleveland_entropy_forests, "EntropyCleveland_train")
	create_graph("Cleveland Data (Split Criterion = Gini)", "# of Features", "Accuracy", cleveland_gini_forests, "GiniCleveland_train")
	create_graph("Banknote Data (Split Criterion = Entropy)", "# of Features", "Accuracy", banknote_entropy_forests, "EntropyBanknote_train")
	create_graph("Banknote Data (Split Criterion = Gini)", "# of Features", "Accuracy", banknote_gini_forests, "GiniBanknote_train")

main()