#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd 
import matplotlib.pyplot as plt
from sklearn import datasets 
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn.metrics import classification_report
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler


# In[4]:


comp = pd.read_csv("C:/Users/DELL/Downloads/Company_Data (1).csv")


# In[5]:


comp


# In[6]:


comp.head()


# In[7]:


comp.shape


# In[8]:


comp.info()


# In[11]:


comp['ShelveLoc'] = comp['ShelveLoc'].astype('category')
comp['Urban'] = comp['Urban'].astype('category')
comp['US'] =comp['US'].astype('category')


# In[12]:


comp.dtypes


# In[13]:


sales_mean = comp.Sales.mean()
sales_mean


# In[15]:


comp['High'] = comp.Sales.map(lambda x: 1 if x > 8 else 0) # mean = 7.49 is round of 8
comp.High


# In[16]:


comp.head()


# In[18]:


label_encoder = preprocessing.LabelEncoder()
comp['ShelveLoc'] = label_encoder.fit_transform(comp['ShelveLoc'])


# In[20]:


comp['Urban'] = label_encoder.fit_transform(comp['Urban'])
comp['US'] = label_encoder.fit_transform(comp['US'])


# In[21]:


comp


# In[22]:


x = comp.iloc[:,1:11]
y = comp['High']


# In[23]:


x


# In[24]:


y


# In[25]:


comp['High'].unique()


# In[26]:


comp.High.value_counts()


# In[27]:


x_train,x_test,y_train,y_test = train_test_split(x,y,test_size =0.2,random_state = 0)


# In[29]:


model = DecisionTreeClassifier(criterion = 'entropy',max_depth = 3,class_weight = 'balanced') #model building by C5.0)


# In[30]:


model.fit(x_train,y_train)


# In[33]:


fn = ['CompPrice','Income','Advertising','Population','Price','ShelveLoc','Age','Education','Urban','US']# we will extract the features
cn = ['Sales is high','Sales is low']
fig,axes = plt.subplots(nrows = 1 ,ncols = 1,figsize =(5,5),dpi = 300)
tree.plot_tree(model,feature_names = fn,class_names = cn,filled = True);


# In[35]:


preds = model.predict(x_test)
pd.Series(preds).value_counts()


# In[36]:


preds


# In[37]:


crosstable = pd.crosstab(y_test,preds)
crosstable


# In[38]:


np.mean(preds==y_test)


# In[40]:


print(classification_report(preds,y_test))


# In[41]:


model_1 = DecisionTreeClassifier(criterion = 'gini',max_depth = 3 ,class_weight = 'balanced')# model building by CART


# In[42]:


model_1.fit(x_train,y_train)


# In[43]:


tree.plot_tree(model_1)


# In[45]:


preds1 = model_1.predict(x_test)
preds1


# In[46]:


pd.Series(preds1).value_counts()


# In[47]:


np.mean(preds1 == y_test)


# # By Bagging and Boosting

# In[48]:


from sklearn.metrics import accuracy_score#importing metrics for accuracy calculation (confusion matrix)
from sklearn.ensemble import BaggingClassifier#bagging combines the results of multipls models to get a generalized result. 
from sklearn.ensemble import AdaBoostClassifier #boosting method attempts to correct the errors of previous models.
#from sklearn.tree import export_graphviz
#from sklearn.externals.six import StringIO
#from IPython.display import Image
#from pydot import graph_from_dot_data
from sklearn.metrics import classification_report, confusion_matrix


# In[49]:


dcmodel =  BaggingClassifier(DecisionTreeClassifier(max_depth = 6), random_state=0) #decision tree classifier object
dcmodel =  AdaBoostClassifier(DecisionTreeClassifier(max_depth = 6), random_state=0) #decision tree classifier object


# In[50]:


cmodel = dcmodel.fit(x_train,y_train) #train decision tree
y_predict = dcmodel.predict(x_test)


# In[51]:


print("Accuracy : ", accuracy_score(y_test,y_predict)*100 )


# In[52]:


print(confusion_matrix(y_test,y_predict))
print(classification_report(y_test,y_predict))


# In[ ]:




