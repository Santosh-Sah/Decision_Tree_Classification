# -*- coding: utf-8 -*-
"""
Created on Mon Mar 16 21:45:18 2020

@author: Santosh Sah
"""

from DecisionTreeClassificationUtils import (importDecisionTreeClassificationDataset, saveTrainingAndTestingDataset)

def preprocess():
    
    X_train, X_test, y_train, y_test = importDecisionTreeClassificationDataset("Decision_Tree_Classification_Social_Network_Ads.csv")
    
    saveTrainingAndTestingDataset(X_train, X_test, y_train, y_test)
    

if __name__ == "__main__":
    preprocess()