#import library
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#import dataset 
#dataset = pd.read_csv('Position_Salaries.csv')
dataset=pd.read_csv('E:\Machine_Learning_A_Z\Decission_tree\Position_Salaries.csv')
X=dataset.iloc[:,1:2].values
y=dataset.iloc[:,2].values
#Fitting the Decision Tree Regression to the dataset
from sklearn.tree import DecisionTreeRegressor
regressor=DecisionTreeRegressor(random_state=0)
regressor.fit(X,y)

#Predicting a new result
y_pred=regressor.predict(np.array(6.5).reshape(1,1))

#Visualising the Decision tree regression results
plt.scatter(X,y,color='red')
plt.plot(X,regressor.predict(X),color='blue')
plt.title('Truth or Bluff (Decision Tree Regression)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()

#Visualizating the Regressor results (for higher resolution and smoother curve)
X_grid=np.arange(min(X),max(X),0.1)
X_grid=X_grid.reshape(len(X_grid),1)
plt.scatter(X,y,color='red')
plt.plot(X_grid,regressor.predict(X_grid),color='blue')
plt.title('Truth or Bluff (Decission Tree Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()



