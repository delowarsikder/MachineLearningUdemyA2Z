#import library
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#import dataset
dataset=pd.read_csv('E:\Machine_Learning_A_Z\Apriori\Market_Basket_Optimisation.csv',header=None)
transactions=[] #create a list
for i in range(0,7501):
    transactions.append([str(dataset.values[i,j]) for j in range(0,20)])  #add data of different customer

#Training Aprioi on the dataset
from apyori import apriori
rules=apriori(transactions,min_support=0.003,min_confidence=0.2,min_lift=3,min_length=2)

#Visualization the results
results=list(rules)