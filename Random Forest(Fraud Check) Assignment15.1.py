#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn.metrics import classification_report
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score


# In[3]:


fraud = pd.read_csv("C:/Users/Administrator/Downloads/Fraud_check.csv ")


# In[4]:


fraud


# In[5]:


fraud.info()


# In[6]:


fraud.shape


# In[7]:


# Renaming the columns 

fraud.rename({'Undergrad':'UG','Marital.Status':'MS', 'Taxable.Income':'TI', 'City.Population':'CP', 'Work.Experience':'WE'},axis = 1, inplace = True)


# In[9]:


# Categorizing the tax column based on the condition

fraud['TI'] = fraud.TI.map(lambda taxable_income : 'Risky' if taxable_income <= 30000 else 'Good')


# In[10]:


fraud.head()


# In[11]:


# Converting the categorical columns to proper datatypes

fraud['UG'] = fraud['UG'].astype("category")
fraud['MS'] = fraud['MS'].astype("category")
fraud['Urban'] = fraud['Urban'].astype("category")
fraud['TI'] = fraud['TI'].astype("category")


# In[12]:


fraud.dtypes


# In[13]:


# Encoding the categorical columns by using label encoder

label_encoder = preprocessing.LabelEncoder()
fraud['UG'] = label_encoder.fit_transform(fraud['UG'])
fraud['MS'] = label_encoder.fit_transform(fraud['MS'])
fraud['Urban'] = label_encoder.fit_transform(fraud['Urban'])
fraud['TI'] = label_encoder.fit_transform(fraud['TI'])


# In[14]:


fraud


# In[15]:


fraud['TI'].unique()


# In[16]:


fraud['TI'].value_counts()


# In[17]:


# Splitting the data into x and y as input and output
x= fraud.iloc[:,[0,1,3,4,5]]
y= fraud.iloc[:,2]


# In[22]:


x


# In[19]:


y


# In[24]:


# Splitting the data into training and test dataset

x_train,x_test,y_train,y_test = train_test_split(x,y, test_size = 0.3, random_state = 10)


# ## Random Forest

# In[25]:


from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_jobs=3,oob_score=True,n_estimators=15,max_features=4,criterion="entropy")


# In[26]:


rf.fit(x_train,y_train) # Fitting RandomForestClassifier model from sklearn.ensemble 
rf.estimators_ # 
rf.classes_ # class labels (output)
rf.n_classes_ # Number of levels in class labels 
rf.n_features_  # Number of input features in model 8 here.
rf.n_outputs_ # Number of outputs when fit performed


# In[27]:


preds = rf.predict(x_test)
preds


# In[28]:


pd.Series(preds).value_counts()


# In[29]:


crosstable = pd.crosstab(preds,y_test)
crosstable


# In[30]:


# Final step we will calculate the accuracy of our model
# We are comparing the predicted values with the actual values and calculating mean for the matches
np.mean(preds==y_test)


# In[31]:


print(classification_report(preds,y_test))


# In[ ]:




