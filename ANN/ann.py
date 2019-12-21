# -*- coding: utf-8 -*-
"""
Created on Mon Oct 14 18:52:45 2019

@author: DelowaR
"""
#this accuracy don't match with tutorial which is not understood

#part -1 -Data preprocessing
#import library
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#import dataset
dataset=pd.read_csv('E:\Machine_Learning_A_Z\ANN\Churn_Modelling.csv')
X=dataset.iloc[:,3:13].values
y=dataset.iloc[:,13].values

#Encoding Categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
label_encoder_X_1=LabelEncoder()
X[:,1]=label_encoder_X_1.fit_transform(X[:,1])
label_encoder_X_2=LabelEncoder()
X[:,2]=label_encoder_X_2.fit_transform(X[:,2])
oneHotenconder=OneHotEncoder(categorical_features=[1])
X=oneHotenconder.fit_transform(X).toarray()
X=X[:,1:]

#Spliting the dataset into the training and test set
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=0)

#Feature Scaling
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
X_train=sc.fit_transform(X_train)
X_test=sc.transform(X_test)

#Part -2  make the ANN

#import the keras libraries and packages
import keras
from keras.models import Sequential
from keras.layers import Dense

#Initialing the ANN
classifier = Sequential()

#Adding the input layer and the first hidden layer
classifier.add(Dense(output_dim=6,init='uniform',activation='relu',input_dim=11))
#input_dim is input which is independent variable  
#output_dim no of node
#output_dim is output binary 0 or 1 and input element is 11 so average (11+1)/2

#Adding second hidden layer
classifier.add(Dense(output_dim=6,init='uniform',activation='relu'))

#Adding the output layer
classifier.add(Dense(output_dim=1,init='uniform',activation='sigmoid'))
#sigmoid has binary outcome
#Compiling the ANN
classifier.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
 
#Fititng the ANN  to the traing set 
classifier.fit(X_train,y_train,batch_size=10,nb_epoch=100)


#Part -3
#making the predictions and evaluating the model
#predicting the test set results

y_pred=classifier.predict(X_test)
y_pred=(y_pred>0.5)
#making the confusion matrix
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_pred)

