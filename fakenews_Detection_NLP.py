# -*- coding: utf-8 -*-
"""
Created on Fri May 29 11:00:40 2020

@author: sumanth
"""
import pandas as pd

#Read the data
text=pd.read_csv('news.csv')
#Get shape and head
text.shape
text.head()

############################################################################ 


#Get the labels
labels=text.label
labels.head()

X=text['text']
Y=labels

#Split the dataset
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(X,Y, test_size=0.3, random_state=23)
x_train.shape,x_test.shape,y_train.shape,y_test.shape

##############################################################################

# ########################33TfidfVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

tfidf=TfidfVectorizer(stop_words='english', max_df=0.7)


#### fit and transform train set,transform test set

tfidf_train=tfidf.fit_transform(x_train)
tfidf_test=tfidf.transform(x_test)

from sklearn.linear_model import PassiveAggressiveClassifier

ac=PassiveAggressiveClassifier(max_iter=100).fit(tfidf_train,y_train)

y_pred=ac.predict(tfidf_test)



from sklearn.metrics import accuracy_score, confusion_matrix
Confusion=confusion_matrix(y_pred,y_test)
print(Confusion)

Accuracy=accuracy_score(y_pred,y_test)
print("Accuracy :",(Accuracy*100).round(3))


ACC=[]
for i in range(20,50):
    acc=PassiveAggressiveClassifier(max_iter=i).fit(tfidf_train,y_train)
    ACC.append((acc.score(tfidf_test,y_test)*100).round(2))
    
print(max(ACC))





############################################################################ 


from sklearn.linear_model import LogisticRegression

log=LogisticRegression().fit(tfidf_train,y_train)
L_pred=log.predict(tfidf_test)

Acc=accuracy_score(L_pred,y_test)
print("Accuracy : ",Acc*100)