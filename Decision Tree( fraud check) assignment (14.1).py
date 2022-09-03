#!/usr/bin/env python
# coding: utf-8

# # Decision Tree(fraud check)

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets  
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import  DecisionTreeClassifier
from sklearn import tree
from sklearn.metrics import classification_report
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler


# In[2]:


fraud=pd.read_csv("C:/Users/DELL/Downloads/Fraud_check.csv")


# In[3]:


fraud 


# In[4]:


fraud.head()


# In[5]:


fraud.info()


# In[6]:


fraud.rename({'Undergrad':'UG','Marital.Status':'MS', 'Taxable.Income':'TI', 'City.Population':'CP', 'Work.Experience':'WE'},axis = 1, inplace = True)


# In[7]:


fraud.head()


# In[8]:


# Categorizing the tax column based on the condition

fraud['TI'] = fraud.TI.map(lambda taxable_income : 'Risky' if taxable_income <= 30000 else 'Good')


# In[9]:


# Converting the categorical columns to proper datatypes

fraud['UG'] = fraud['UG'].astype("category")
fraud['MS'] = fraud['MS'].astype("category")
fraud['Urban'] = fraud['Urban'].astype("category")
fraud['TI'] = fraud['TI'].astype("category")


# In[10]:


fraud.info()


# In[11]:


fraud.head() #TI in catagorical


# In[12]:


# Encoding the categorical columns by using label encoder

label_encoder = preprocessing.LabelEncoder()
fraud['UG'] = label_encoder.fit_transform(fraud['UG'])

fraud['MS'] = label_encoder.fit_transform(fraud['MS'])

fraud['Urban'] = label_encoder.fit_transform(fraud['Urban'])

fraud['TI'] = label_encoder.fit_transform(fraud['TI'])


# In[13]:


fraud


# In[14]:


fraud['TI'].unique()


# In[15]:


fraud['TI'].value_counts()


# In[16]:


# Splitting the data into x and y as input and output

x= fraud.iloc[:,[0,1,3,4,5]]
y= fraud.iloc[:,2]


# In[17]:


x


# In[18]:


y


# In[19]:


# Splitting the data into training and test dataset

x_train,x_test,y_train,y_test = train_test_split(x,y, test_size = 0.3, random_state = 10)


# In[20]:


model = DecisionTreeClassifier(criterion = 'entropy', max_depth = 3, class_weight = 'balanced') #model build by C5.0
model.fit(x_train,y_train)


# In[21]:


tree.plot_tree(model)


# In[22]:


fn = ['Undergrad',	'Marital.Status',	'City.Population',	'Work.Experience',	'Urban']
cn = ['Taxable_income is Risky', 'Taxable_income is Good']
fig,axes = plt.subplots(nrows = 1, ncols =1, figsize =(4,4), dpi = 300)   
tree.plot_tree(model, feature_names = fn, class_names = cn, filled = True);


# In[23]:


preds = model.predict(x_test)
preds


# In[24]:


pd.Series(preds).value_counts()


# In[25]:


crosstable = pd.crosstab(preds,y_test) #confusion matrics
crosstable


# In[26]:


np.mean(preds==y_test)


# In[27]:


print(classification_report(preds,y_test))


# In[28]:


model_cart = DecisionTreeClassifier(criterion = 'gini', max_depth = 3, class_weight = 'balanced') #build model by CART
model_cart.fit(x_train,y_train)


# In[29]:


tree.plot_tree(model_cart)


# In[30]:


preds1 = model_cart.predict(x_test)
preds1


# In[31]:


np.mean(preds1==y_test)


# In[32]:


from sklearn.metrics import f1_score
print(f1_score(preds1,y_test))


# # By Bagging Boosting

# In[33]:


from sklearn.metrics import accuracy_score#importing metrics for accuracy calculation (confusion matrix)
from sklearn.ensemble import BaggingClassifier#bagging combines the results of multipls models to get a generalized result. 
from sklearn.ensemble import AdaBoostClassifier #boosting method attempts to correct the errors of previous models.
#from sklearn.tree import export_graphviz
#from sklearn.externals.six import StringIO
#from IPython.display import Image
#from pydot import graph_from_dot_data
from sklearn.metrics import classification_report, confusion_matrix


# In[34]:


dcmodel =  BaggingClassifier(DecisionTreeClassifier(max_depth = 6), random_state=0) #decision tree classifier object
dcmodel =  AdaBoostClassifier(DecisionTreeClassifier(max_depth = 6), random_state=0) #decision tree classifier object


# In[35]:


dcmodel = dcmodel.fit(x_train,y_train) #train decision tree
y_predict = dcmodel.predict(x_test)


# In[36]:


print("Accuracy : ", accuracy_score(y_test,y_predict)*100 )


# In[37]:


print(confusion_matrix(y_test,y_predict))
print(classification_report(y_test,y_predict))

