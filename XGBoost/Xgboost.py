# -*- coding: utf-8 -*-
"""
Created on Thu Oct 24 00:34:12 2019

@author: DelowaR
"""
#install xgboost 
#open anaconda command run # conda install -c anaconda py-xgboost
#open git bash # git clone --recursive https://github.com/dmlc/xgboost

#import library
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#import dataset
dataset =pd.read_csv('E:\Machine_Learning_A_Z\XGBoost\Churn_Modelling.csv');
X=dataset.iloc[:,3:13].values
y=dataset.iloc[:,13].values

#Encoding Categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
label_encoder_X_1=LabelEncoder()
X[:,1]=label_encoder_X_1.fit_transform(X[:,1])
label_encoder_X_2=LabelEncoder()
X[:,2]=label_encoder_X_2.fit_transform(X[:,2])
onehotenconder=OneHotEncoder(categorical_features=[1])
X=onehotenconder.fit_transform(X).toarray()
X=X[:,1:]

#Spliting a dataset into the Training set and test set
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=0)

#Fitting XGBoost to the Traing set
from xgboost import XGBClassifier
classifier=XGBClassifier()
classifier.fit(X_train,y_train)

#Predicting the Test set results
y_pred=classifier.predict(X_test)
#Making the confusing metrics
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_pred)

#Applying the k-fold cross validation
from sklearn.model_selection import cross_val_score
accuracies=cross_val_score(estimator=classifier,X=X_train,y=y_train,cv=10)
accuracies.mean()
accuracies.std() 










