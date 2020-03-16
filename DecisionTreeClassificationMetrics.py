# -*- coding: utf-8 -*-
"""
Created on Mon Mar 16 21:52:26 2020

@author: Santosh Sah
"""

from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from DecisionTreeClassificationUtils import (readDecisionTreeClassificationYTest, readDecisionTreeClassificationYPred)

"""

calculating DecisionTreeClassification confussion matrix

"""
def testDecisionTreeClassificationConfussionMatrix():
    
    y_test = readDecisionTreeClassificationYTest()
    y_pred = readDecisionTreeClassificationYPred()
    
    decisionTreeClassificationConfussionMatrix = confusion_matrix(y_test, y_pred)
    print(decisionTreeClassificationConfussionMatrix)
    
    """
    Below is the confussion matrix
    [[52  6]
    [ 3 19]]
    
    """
"""
calculating accuracy score

"""

def testDecisionTreeClassificationAccuracy():
    
    y_test = readDecisionTreeClassificationYTest()
    y_pred = readDecisionTreeClassificationYPred()
    
    decisionTreeClassificationConfussionAccuracy = accuracy_score(y_test, y_pred)
    
    print(decisionTreeClassificationConfussionAccuracy) #.8875%

"""
calculating classification report

"""

def testDecisionTreeClassificationClassificationReport():
    
    y_test = readDecisionTreeClassificationYTest()
    y_pred = readDecisionTreeClassificationYPred()
    
    decisionTreeClassificationConfussionClassificationReport = classification_report(y_test, y_pred)
    
    print(decisionTreeClassificationConfussionClassificationReport)
    
    """
               precision    recall  f1-score   support

          0       0.95      0.90      0.92        58
          1       0.76      0.86      0.81        22

avg / total       0.89      0.89      0.89        80
    """
    
if __name__ == "__main__":
    #testDecisionTreeClassificationConfussionMatrix()
    #testDecisionTreeClassificationAccuracy()
    testDecisionTreeClassificationClassificationReport()