#!/usr/bin/env python
# coding: utf-8

# In[32]:


import pandas as pd 
import numpy as np
import seaborn as sns
from sklearn.metrics import classification_report
from sklearn.naive_bayes import GaussianNB


# In[33]:


from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, cross_val_score


# In[34]:


test=pd.read_csv("C:/Users/Administrator/Downloads/SalaryData_Test.csv")
train=pd.read_csv("C:/Users/Administrator/Downloads\SalaryData_Train.csv")


# In[35]:


test.head()


# In[36]:


train.head()


# In[37]:


test.shape


# In[38]:


train.shape


# In[39]:


test.info()


# In[40]:


train.info()


# In[41]:


#Converting dtypes for train
train['workclass']=train['workclass'].astype('category')
train['education']=train['education'].astype('category')
train['maritalstatus']=train['maritalstatus'].astype('category')
train['occupation']=train['occupation'].astype('category')
train['relationship']=train['relationship'].astype('category')
train['race']=train['race'].astype('category')
train['native']=train['native'].astype('category')
train['sex']=train['sex'].astype('category')


# In[42]:


train.info()


# In[43]:


from sklearn import preprocessing                      
label_encoder = preprocessing.LabelEncoder()


# In[44]:


train['workclass'] = label_encoder.fit_transform(train['workclass'])
train['education'] = label_encoder.fit_transform(train['education'])
train['maritalstatus'] = label_encoder.fit_transform(train['maritalstatus'])
train['occupation'] = label_encoder.fit_transform(train['occupation'])
train['relationship'] = label_encoder.fit_transform(train['relationship'])
train['race'] = label_encoder.fit_transform(train['race'])
train['sex'] = label_encoder.fit_transform(train['sex'])
train['native'] = label_encoder.fit_transform(train['native'])


# In[45]:


train.head()


# In[46]:


test['workclass']=test['workclass'].astype('category')
test['education']=test['education'].astype('category')
test['maritalstatus']=test['maritalstatus'].astype('category')
test['occupation']=test['occupation'].astype('category')
test['relationship']=test['relationship'].astype('category')
test['race']=test['race'].astype('category')
test['native']=test['native'].astype('category')
test['sex']=test['sex'].astype('category')


# In[47]:


test.info()


# In[48]:


test['workclass'] = label_encoder.fit_transform(test['workclass'])
test['education'] = label_encoder.fit_transform(test['education'])
test['maritalstatus'] = label_encoder.fit_transform(test['maritalstatus'])
test['occupation'] = label_encoder.fit_transform(test['occupation'])
test['relationship'] = label_encoder.fit_transform(test['relationship'])
test['race'] = label_encoder.fit_transform(test['race'])
test['sex'] = label_encoder.fit_transform(test['sex'])
test['native'] = label_encoder.fit_transform(test['native'])


# In[49]:


test.head()


# In[50]:


train['Salary'] = label_encoder.fit_transform(train['Salary'])
test['Salary'] = label_encoder.fit_transform(test['Salary'])


# In[51]:


test.head()


# In[52]:


train.head()


# In[53]:


# # Splitting the data into x train y train and x test y test 


# In[54]:


trainx=train.iloc[:,0:13]
trainy=train.iloc[:,13]
testx=test.iloc[:,0:13]
testy=test.iloc[:,13]


# In[55]:


trainx.shape ,trainy.shape, testx.shape, testy.shape


# In[56]:


trainx


# In[57]:


trainy


# In[58]:


# train a Gaussian Naive Bayes classifier on the training set
from sklearn.naive_bayes import GaussianNB


# In[59]:


gnb = GaussianNB()


# In[60]:


gnb.fit(trainx,trainy)


# In[61]:


y_pred = gnb.predict(testx)

y_pred


# In[62]:


#comparing train set and test set accuracing 
acc = accuracy_score(testy, y_pred) * 100
print("Accuracy =", acc)


# In[63]:


print('Training set score: {:.3f}'.format(gnb.score(trainx, trainy)))

print('Test set score: {:.3f}'.format(gnb.score(testx, testy)))


# In[64]:


# # confusion matrix


# In[65]:


pd.crosstab(y_pred,testy)


# In[66]:


y_pred


# In[67]:


from sklearn.metrics import confusion_matrix

cm = confusion_matrix(testy, y_pred)

print('Confusion matrix\n\n', cm)

print('\nTrue Positives(TP) = ', cm[0,0])

print('\nTrue Negatives(TN) = ', cm[1,1])

print('\nFalse Positives(FP) = ', cm[0,1])

print('\nFalse Negatives(FN) = ', cm[1,0])


# In[68]:


cm_matrix = pd.DataFrame(data=cm, columns=['Actual Positive:1', 'Actual Negative:0'], 
                                 index=['Predict Positive:1', 'Predict Negative:0'])

sns.heatmap(cm_matrix, annot=True, fmt='d', cmap='YlGnBu')


# In[ ]:





# In[ ]:




