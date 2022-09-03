#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from sklearn import preprocessing 
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import seaborn as sns
from sklearn.linear_model import LogisticRegression


# In[2]:


bank = pd.read_csv("C:/Users/DELL/Downloads/bank-full (2).csv",sep=';')


# In[3]:


bank


# In[4]:


bank.dropna()


# In[5]:


print(bank.shape)


# In[6]:


print(list(bank.columns))


# In[7]:


# barplot for y
sns.countplot(x='y',data=bank,palette='hls') 
plt.show()


# In[8]:


bank.isnull().sum()


# In[9]:


sns.countplot(y = "job",data=bank)
plt.show()


# In[10]:


# Customer marital status distribution
sns.countplot(y="marital",data=bank)


# In[11]:


#Credit in Default
sns.countplot(y="default",data=bank)
plt.show()


# In[12]:


#Hosing Loan
sns.countplot(y="housing",data = bank)
plt.show()


# In[13]:


#Personal Loan
sns.countplot(y="loan",data= bank)
plt.show()


# In[14]:


#Barplot for previous marketing campaign outcome
sns.countplot(y="poutcome",data = bank)
plt.show()


# In[15]:


bank.drop(bank.columns[[0,3,7,8,9,10,11,12,13,14,]],axis=1,)


# In[16]:


bank2 = pd.get_dummies(bank,columns = ['job','marital','default','housing','loan','poutcome'])


# In[17]:


bank2


# In[18]:


bank2.drop(bank2.columns[[12,16,18,21,24]],axis=1,inplace=True)


# In[19]:


bank2.columns


# In[20]:


sns.heatmap(bank2.corr())
plt.show()


# In[21]:


X = bank2.iloc[:,11:]
y = bank2.iloc[:,10]
X_train,X_test,y_train,y_test = train_test_split(X,y,random_state = 0)


# In[22]:


X_train.shape


# In[23]:


classifier = LogisticRegression(random_state=0)
classifier.fit(X_train,y_train)


# In[24]:


y_pred = classifier.predict(X_test)
from sklearn.metrics import confusion_matrix
confusion_matrix = confusion_matrix(y_test,y_pred)
print(confusion_matrix)


# In[25]:


print('Accuracy of logistic regression classifier on test set:{:.2f}'.format(classifier.score(X_test, y_test)))

