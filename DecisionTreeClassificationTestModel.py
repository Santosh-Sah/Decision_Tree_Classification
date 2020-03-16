# -*- coding: utf-8 -*-
"""
Created on Mon Mar 16 21:06:14 2020

@author: Santosh Sah
"""

from DecisionTreeClassificationUtils import (readDecisionTreeClassificationXTest, readDecisionTreeClassificationModel,
                                     saveDecisionTreeClassificationYPred, readDecisionTreeClassificationStandardScaler)

"""
test the model on testing dataset
"""
def testDecisionTreeClassificationModel():
    
    X_test = readDecisionTreeClassificationXTest()
    decisionTreeClassificationStandardScaler = readDecisionTreeClassificationStandardScaler()
    X_test = decisionTreeClassificationStandardScaler.transform(X_test)
    
    decisionTreeClassificationModel = readDecisionTreeClassificationModel()
    
    y_pred = decisionTreeClassificationModel.predict(X_test)
    saveDecisionTreeClassificationYPred(y_pred)
    
    print(y_pred)
    
if __name__ == "__main__":
    testDecisionTreeClassificationModel()