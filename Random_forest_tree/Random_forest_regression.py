# -*- coding: utf-8 -*-
"""
Created on Tue Sep 24 15:29:33 2019

@author: DelowaR
"""

#import library
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#import dataset
dataset=pd.read_csv('E:\Machine_Learning_A_Z\Random_forest_tree\Position_Salaries.csv')
X=dataset.iloc[:,1:2].values
y=dataset.iloc[:,2].values

#Feature Scalling
#Fitting Random forest Regressor to the dataset
from sklearn.ensemble import RandomForestRegressor
regressor=RandomForestRegressor(n_estimators=300,random_state=0)
regressor.fit(X,y)

#Predicting a new result
y_pred=regressor.predict([[6.5]])

#Visualizing the regression results (for higher resolution and smother curve)
X_grid=np.arange(min(X),max(X),0.01)
X_grid=X_grid.reshape((len(X_grid),1))
plt.scatter(X,y,color='red')
plt.plot(X_grid,regressor.predict(X_grid),color='blue')
plt.title('Truth or Bluff (Random Forest Regressor)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()













