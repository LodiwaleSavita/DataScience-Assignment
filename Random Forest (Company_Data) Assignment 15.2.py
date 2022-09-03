#!/usr/bin/env python
# coding: utf-8

# ## Random Forest 

# In[1]:


#import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn import preprocessing    


# In[2]:


company=pd.read_csv("C:/Users/Administrator/Downloads/Company_Data.csv")


# In[3]:


company


# In[4]:


company.head()


# In[5]:


company.info()


# In[6]:


company.shape


# In[7]:


# As the three columns ShelveLoc, Urban, US are of object type need to convert that in proper categorical columns
company['ShelveLoc'] = company['ShelveLoc'].astype('category')
company['Urban'] = company['Urban'].astype('category')
company['US'] = company['US'].astype('category')


# In[8]:


company.dtypes


# In[9]:


# As we have to convert the sales column into categorical column so need to calculate the mean first and then using lambda function categorize that column into 0's and 1's according to the mean
sales_mean = company.Sales.mean()
sales_mean


# In[10]:


company['High'] = company.Sales.map(lambda x: 1 if x > 8 else 0)


# In[11]:


company.High


# In[12]:


company


# In[13]:


label_encoder = preprocessing.LabelEncoder()
company['ShelveLoc'] = label_encoder.fit_transform(company['ShelveLoc'])
company['Urban'] = label_encoder.fit_transform(company['Urban'])
company['US'] = label_encoder.fit_transform(company['US'])


# In[14]:


# Splitting the data into x and y as input and output
x= company.iloc[:,1:11]
y= company['High']


# In[15]:


x


# In[16]:


y


# In[17]:


# Displaying the unique values, there are only two unique values 0 and 1, for high sales(sales>8) it is 1 and for low sales(sales<8) it is 0

company['High'].unique()


# In[18]:


company.High.value_counts()


# In[19]:


#Splitting the data into training and test dataset
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.2, random_state=0)


# In[20]:


## Random Forest


# In[21]:


rf = RandomForestClassifier(n_jobs=3,oob_score=True,n_estimators=15,criterion="entropy")


# In[22]:


rf.fit(x_train,y_train) # Fitting RandomForestClassifier model from sklearn.ensemble 
rf.estimators_ # 
rf.classes_ # class labels (output)
rf.n_classes_ # Number of levels in class labels 
rf.n_features_  # Number of input features in model 8 here.

rf.n_outputs_ # Number of outputs when fit performed

rf.oob_score_ 


# In[23]:


rf.predict(x_test)


# In[24]:


preds = rf.predict(x_test)
pd.Series(preds).value_counts()


# In[25]:


preds


# In[26]:


# In order to check whether the predictions are correct or wrong we will create a cross tab on y_test data

crosstable = pd.crosstab(y_test,preds)
crosstable


# In[27]:


# Final step we will calculate the accuracy of our model

# We are comparing the predicted values with the actual values and calculating mean for the matches
np.mean(preds==y_test)


# In[28]:


print(classification_report(preds,y_test))


# In[ ]:




