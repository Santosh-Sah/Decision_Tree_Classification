# -*- coding: utf-8 -*-
"""
Created on Mon Mar 16 21:49:22 2020

@author: Santosh Sah
"""

import pandas as pd
from DecisionTreeClassificationUtils import readDecisionTreeClassificationModel, readDecisionTreeClassificationStandardScaler

def predict():
    
    decisionTreeClassification = readDecisionTreeClassificationModel()
    decisionTreeClassificationStandardScaler = readDecisionTreeClassificationStandardScaler()
    
    inputValue = [[26, 1000]]
    inputValueDataframe = pd.DataFrame(decisionTreeClassificationStandardScaler.transform(inputValue))
    
    predictedValue = decisionTreeClassification.predict(inputValueDataframe.values)
    
    print(predictedValue)

if __name__ == "__main__":
    predict()