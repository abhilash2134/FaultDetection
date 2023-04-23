# -*- coding: utf-8 -*-
"""
Created on Fri Janurary 22 09:05:55 2022

@author: Abhilash Nair and Jonas Wietzel

"""


import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.metrics import accuracy_score, confusion_matrix
import pickle

names = [
    "NearestNeighbors",
    "GaussianProcess",
    "DecisionTree",
    "RandomForest",
    "NeuralNet",
    "AdaBoost",
    "NaiveBayes",
    "QDA",
]

classifiers = [
   KNeighborsClassifier(2),
   GaussianProcessClassifier(1.0 * RBF(1.0)),
   DecisionTreeClassifier(max_depth=5), #99.45 erros
   RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
   MLPClassifier(alpha=1, max_iter=1000),
   AdaBoostClassifier(),
   GaussianNB(),
   QuadraticDiscriminantAnalysis(),
]

#%% Loading and pre-preparing data
RUNNING_DIR  = os.path.dirname(os.path.abspath(__file__))
path_to_data = os.path.join(RUNNING_DIR, 'Data','Linnes_labelled_cni.csv')

data_csv = pd.read_csv(path_to_data, low_memory=False, parse_dates=['Data Dimensions/Time'], dayfirst=True)
data_csv = data_csv.sort_values(by='Data Dimensions/Time')
data_csv = data_csv.dropna()
timestamp = data_csv['Data Dimensions/Time'].tolist()

data_csv[['CNI_diff', 'QIN_diff']] = data_csv[['CNI', 'QIN']].diff() #calculate the difference


# Replace tags with '1' and '0'
data_csv['CNI Labelling_numeric'] = data_csv['CNI data cleaning'].replace('Rest', 1)
data_csv['CNI Labelling_numeric'] = data_csv['CNI Labelling_numeric'].replace('CNI outlier',0)

# Converting data to array
X = data_csv[['CNI', 'QIN', 'CNI_diff', 'QIN_diff']].to_numpy()[1:-1]
y = data_csv['CNI Labelling_numeric'].to_numpy()[1:-1]

# Split data to traning and test data
split = len(X) // 4
X_train, X_test, y_train, y_test = X[split:], X[:split], y[split:], y[:split]
i = 1


#%% Train Classification models
for name, clf in zip(names, classifiers):
    try:
        
        print(name)
        
        clf.fit(X_train, y_train)
        score = clf.score(X_train, y_train)
        
        # Plot the validation dataset
        y_predict   = clf.predict(X_test)
        
        if hasattr(clf, "decision_function"):
            y_pred_prob = clf.decision_function(X_test)
        else:
            y_pred_prob = clf.predict_proba(X_test)
        
        accuracy    = accuracy_score(y_test, y_predict)
        cm          = confusion_matrix(y_test, y_predict)
        
        false_positive = 100*cm[0,1]/np.count_nonzero(y_test==0) 
        false_negative = 100*cm[1,0]/np.count_nonzero(y_test==1) 
                
        print("Accuracy of validation for {} is : {}%".format(name, accuracy*100))
        print("undetected errors = {} %".format(false_negative))
        print('false positives = {} %'.format(false_positive))
        
    except Exception as e:
        print('Data training Failed',e)
        
    # plot values 
    fig, axes = plt.subplots(2, 1, figsize=(18, 6))
    axes[0].scatter(range(0, len(X_test[:,0])), X_test[:,0], c=y_test, s=1, label='Real')
    axes[0].set_title(name+'  Accuracy = '+str(round(accuracy,3))+' %')
    axes[0].legend()
    axes[1].scatter(range(0, len(X_test[:,0])), X_test[:,0], c=y_predict, s=1, label='Model')
    axes[1].legend()
    plt.show()
        
    # Save trained model
    path_to_model = os.path.join(RUNNING_DIR, 'Models', name)
    pickle.dump(clf, open(path_to_model, 'wb'))
       
    i += 1
