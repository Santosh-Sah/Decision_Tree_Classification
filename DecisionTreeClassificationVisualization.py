# -*- coding: utf-8 -*-
"""
Created on Mon Mar 16 21:14:05 2020

@author: Santosh Sah
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from DecisionTreeClassificationUtils import (readDecisionTreeClassificationModel, readDecisionTreeClassificationXTrain, readDecisionTreeClassificationYTrain,
                                         readDecisionTreeClassificationXTest, readDecisionTreeClassificationYTest)
"""
Visualizing training set results
"""
def visualisingTrainingSetResult():
    X_train = readDecisionTreeClassificationXTrain()
    y_train = readDecisionTreeClassificationYTrain()
    classifier = readDecisionTreeClassificationModel()
    
    # Visualising the Training set results
    X_set, y_set = X_train, y_train
    X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.1),
                         np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.1))
    plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
                 alpha = 0.75, cmap = ListedColormap(('red', 'green')))
    plt.xlim(X1.min(), X1.max())
    plt.ylim(X2.min(), X2.max())
    for i, j in enumerate(np.unique(y_set)):
        plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                    c = ListedColormap(('red', 'green'))(i), label = j)
    plt.title('DecisionTree Classification (Training set)')
    plt.xlabel('Age')
    plt.ylabel('Estimated Salary')
    plt.legend()
    
    plt.savefig("decision_tree_classification_trainingsetresult.png")
    
    plt.show()

"""
Visualizing testing set results
"""
def visualisingTestingSetResult():
    X_test = readDecisionTreeClassificationXTest()
    y_test = readDecisionTreeClassificationYTest()
    classifier = readDecisionTreeClassificationModel()
    
    # Visualising the Test set results
    X_set, y_set = X_test, y_test
    X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.1),
                         np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.1))
    plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
                 alpha = 0.75, cmap = ListedColormap(('red', 'green')))
    plt.xlim(X1.min(), X1.max())
    plt.ylim(X2.min(), X2.max())
    for i, j in enumerate(np.unique(y_set)):
        plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                    c = ListedColormap(('red', 'green'))(i), label = j)
    plt.title('DecisionTree Classification (Test set)')
    plt.xlabel('Age')
    plt.ylabel('Estimated Salary')
    plt.legend()
    
    plt.savefig("decision_tree_classification_testingsetresult.png")
    
    plt.show()

if __name__ == "__main__":
    #visualisingTrainingSetResult()
    visualisingTestingSetResult()