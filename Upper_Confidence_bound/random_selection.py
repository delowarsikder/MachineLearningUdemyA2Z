#import library
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#import datasetdataset=pd.read_csv('E:\Machine_Learning_A_Z\\Upper_Confidence_bound\Ads_CTR_Optimisation.csv')

#Implementing Random Selection
import random
N=10000
d=10
ads_selected=[]
total_reward=0
for n in range (0,N):
    ad=random.randrange(d)
    ads_selected.append(ad)
    reward=dataset.values[n,ad]
    total_reward=total_reward+reward
   

#Visualising the results Histogram
#plt.hist(ads_selected,color='red')
plt.hist(ads_selected)
plt.title('Histogram of ads selections')    
plt.xlabel('Ads')
plt.ylabel('Number of times each ad was selected')
plt.show()