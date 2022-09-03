#!/usr/bin/env python
# coding: utf-8

# In[30]:


#SVM Classification
import pandas as pd
import numpy as np
#from sklearn .feature_extraction.text import CountVectorizer,TfidVectorizer
from sklearn.preprocessing import StandardScaler


# In[31]:


from sklearn import svm
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report


# In[32]:


from sklearn.metrics import accuracy_score,confusion_matrix
from sklearn.model_selection import train_test_split,cross_val_score


# In[33]:


salarytrain=pd.read_csv("C:/Users/Administrator/Downloads/SalaryData_Train(1).csv")
salarytest=pd.read_csv("C:/Users/Administrator/Downloads/SalaryData_Test(1).csv")


# In[34]:


salarytrain


# In[36]:


salarytest


# In[11]:


stest.info()


# In[37]:


salarytest.shape


# In[38]:


salarytrain.info()


# In[39]:


salarytrain.shape


# In[40]:


salarytrain.columns
salarytest.columns
stringcol=['workclass','education','maritalstatus','occupation','relationship','race','sex','native']


# In[41]:


from sklearn import preprocessing


# In[42]:


label_encoder=preprocessing.LabelEncoder()
for i in stringcol:
    salarytrain[i] = label_encoder.fit_transform(salarytrain[i])
    salarytest[i] = label_encoder.fit_transform(salarytest[i])


# In[43]:


salarytrain.head()


# In[45]:


salarytest.head()


# In[46]:


#converting Y column in train test both
salarytrain['Salary']=label_encoder.fit_transform(salarytrain['Salary'])


# In[47]:


salarytest['Salary'] = label_encoder.fit_transform(salarytest['Salary'])


# In[48]:


salarytrain.head()


# In[49]:


salarytest.head()


# In[50]:


salarytrainx=salarytrain.iloc[:,0:13]
salarytrainy=salarytrain.iloc[:,13]
salarytestx=salarytest.iloc[:,0:13]
salarytesty=salarytest.iloc[:,13]


# In[51]:


salarytrainx.shape ,salarytrainy.shape ,salarytestx.shape ,salarytesty.shape


# # kernel = rbf

# In[52]:


# by rbf (radial basis function)
model_rbf = SVC(kernel='rbf')


# In[53]:


model_rbf.fit(salarytrainx,salarytrainy)


# In[54]:


train_pred_rbf = model_rbf.predict(salarytrainx)
test_pred_rbf = model_rbf.predict(salarytestx)


# In[55]:


train_rbf_acc=np.mean(train_pred_rbf==salarytrainy)
test_rbf_acc=np.mean(test_pred_rbf==salarytesty)


# In[56]:


train_rbf_acc


# In[57]:


test_rbf_acc


# In[ ]:




