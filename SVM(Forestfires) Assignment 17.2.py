#!/usr/bin/env python
# coding: utf-8

# In[4]:


# SVM Classification
import pandas as pd
import numpy as np
#from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import StandardScaler


# In[5]:


from sklearn import svm
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report


# In[6]:


from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split, cross_val_score


# In[8]:


forest=pd.read_csv("C:/Users/Administrator/Downloads/forestfires.csv")


# In[9]:


forest


# In[10]:


forest.head()


# In[11]:


forest.info()


# In[12]:


forest.shape


# In[13]:


from sklearn import preprocessing


# In[14]:


label_encoder=preprocessing.LabelEncoder()
forest['month'] = label_encoder.fit_transform(forest['month'])
forest['day'] = label_encoder.fit_transform(forest['day'])
forest['size_category'] = label_encoder.fit_transform(forest['size_category'])


# In[15]:


forest


# In[16]:


forest = forest.drop(forest.columns[0:2],axis=1)


# In[17]:


forest.info()


# In[18]:


# Splitting the data into x and y as input and output

x = forest.iloc[:,0:28]
y = forest.iloc[:,28]


# In[19]:


from sklearn.preprocessing import StandardScaler

scaler=StandardScaler()
array_x=scaler.fit_transform(x)


# In[20]:


array_x


# In[21]:


x=pd.DataFrame(array_x)
x


# In[22]:


y


# In[23]:


x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.3, random_state=0)


# In[24]:


x_train.shape, y_train.shape, x_test.shape, y_test.shape


# In[25]:


# # Grid Search CV


# In[26]:


clf = SVC()
param_grid = [{'kernel':['rbf'],'gamma':[60,50,5,10,0.5],'C':[20,15,14,13,12,11,10,0.1,0.001] }]
gsv = GridSearchCV(clf,param_grid,cv=10)
gsv.fit(x_train,y_train)


# In[27]:


gsv.best_params_ , gsv.best_score_


# In[28]:


clf = SVC(C= 20, gamma = 60)
clf.fit(x_train , y_train)
y_pred = clf.predict(x_test)
acc = accuracy_score(y_test, y_pred) * 100
print("Accuracy =", acc)
confusion_matrix(y_test, y_pred)


# In[ ]:




