#Natural Language processing

#import library
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#import dataset
dataset=pd.read_csv('E:\Machine_Learning_A_Z\/NLP\Restaurant_Reviews.tsv',delimiter='\t',quoting=3)
#a=np.asarray(dataset)  convert pandas to numpy array

#convert all data
import re 
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')
from nltk.stem.porter import PorterStemmer
corpus=[]
for i in range(0,1000):
    review=re.sub('[^a-zA-Z]',' ',dataset['Review'][i])#sub function #parameter which don't want to remove 
    review=review.lower() #conver upper case to lower case
    review=review.split()#split functin convert string to list 
    ps=PorterStemmer()
    review=[ps.stem(word) for word in review if not word in set(stopwords.words('english'))] 
        #convert list to string
    review=' '.join(review)
    corpus.append(review)

#Creating the Bag of words model
from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer(max_features=1500)
X=cv.fit_transform(corpus).toarray()
y=dataset.iloc[:,1].values

#Spiling dataset into the Training and Test set
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=.20,random_state=0)

#Fitting classifier to the Training set
from sklearn.naive_bayes import GaussianNB
classifier=GaussianNB()
classifier.fit(X_train,y_train)

#predicting the test set result 
y_pred=classifier.predict(X_test)

#Making the confusing matrix
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_pred)
#Accuracy check 
(55+91)/200

#Cleaning the texts  only first index test
# =============================================================================
# import re 
# import nltk
# from nltk.corpus import stopwords
# nltk.download('stopwords')
# review=re.sub('[^a-zA-Z]',' ',dataset['Review'][0])#sub function #parameter which don't want to remove 
# review=review.lower() #conver upper case to lower case
# review=review.split()#split functin convert string to list 
# review=[word for word in review if not word in set(stopwords.words('english'))]
# 
# #convert root word which key word of sentence ex.loved convert love
# from nltk.stem.porter import PorterStemmer
# ps=PorterStemmer()
# review=[ps.stem(word) for word in review if not word in set(stopwords.words('english'))] 
# #convert list to string
# review=' '.join(review)
# 
# =============================================================================

