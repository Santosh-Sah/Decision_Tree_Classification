# -*- coding: utf-8 -*-
"""
Created on Mon Mar 16 21:39:22 2020

@author: Santosh Sah
"""

from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from DecisionTreeClassificationUtils import (saveDecisionTreeClassificationModel, readDecisionTreeClassificationXTrain, readDecisionTreeClassificationYTrain,
                                     saveDecisionTreeClassificationStandardScaler)

"""
Train DecisionTreeClassification model 
"""
def trainDecisionTreeClassificationModel():
    
    decisionTreeClassificationStandardScalar = StandardScaler()
    
    X_train = readDecisionTreeClassificationXTrain()
    y_train = readDecisionTreeClassificationYTrain()
    
    decisionTreeClassificationStandardScalar.fit(X_train)
    saveDecisionTreeClassificationStandardScaler(decisionTreeClassificationStandardScalar)
    
    X_train = decisionTreeClassificationStandardScalar.transform(X_train)
    
    decisionTreeClassification = DecisionTreeClassifier(criterion = "entropy", random_state = 1234)
    decisionTreeClassification.fit(X_train, y_train)
    
    saveDecisionTreeClassificationModel(decisionTreeClassification)

if __name__ == "__main__":
    trainDecisionTreeClassificationModel()