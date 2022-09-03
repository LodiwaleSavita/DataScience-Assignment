#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib','inline')


# In[4]:


zoo = pd.read_csv("C:/Users/DELL/Downloads/Zoo.csv")


# In[5]:


zoo


# In[6]:


zoo.isnull().sum()


# In[39]:


zoo.info()


# In[40]:


array = zoo.values
X = zoo.iloc[:,1:17]
Y = zoo.iloc[:,17]


# In[41]:


array


# In[42]:


X


# In[43]:


Y


# In[44]:


# Splitting data into training and testing data set
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(X,Y,test_size = 0.30,random_state = 40)


# # KNN CLASSIFICATION

# In[47]:


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)


# In[48]:


from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors = 5,metric = 'minkowski',p =2 )
classifier.fit(x_train,y_train)


# In[49]:


y_pred = classifier.predict(x_test)


# In[50]:


y_test


# In[51]:


y_pred


# In[52]:


from sklearn.metrics import confusion_matrix,accuracy_score
cm = confusion_matrix(y_test,y_pred)
ac = accuracy_score(y_test,y_pred)


# In[53]:


print(cm)


# In[54]:


print(ac)


# In[55]:


n_neighbors = np.array(range(1,50))
param_grid = dict(n_neighbors=n_neighbors)


# In[56]:


model = KNeighborsClassifier()
grid = GridSearchCV(estimator=model,param_grid = param_grid)
grid.fit(X,Y)


# In[57]:


print(grid.best_score_)
print(grid.best_params_)


# In[58]:


k_range = range(1, 4)
k_scores = []


# In[59]:


for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn, X, Y, cv=2)
    k_scores.append(scores.mean())
# plot to see clearly
plt.plot(k_range, k_scores)
plt.xlabel('Value of K for KNN')
plt.ylabel('Cross-Validated Accuracy')
plt.show()


# ### Assignment KNN(Glass)

# In[1]:


import pandas as pd 
import numpy as np
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier


# In[2]:


glass = pd.read_csv("C:/Users/Administrator/Downloads/glass.csv")


# In[3]:


glass


# In[5]:


glass.info()


# In[6]:


glass.isnull().sum()


# In[7]:


array = glass.values
X = glass.iloc[:,0:9]
Y = glass.iloc[:,9]


# In[8]:


X


# In[9]:


Y


# In[10]:


num_folds = 10
kfold = KFold(n_splits=10)


# In[13]:


model = KNeighborsClassifier(n_neighbors=17)
result = cross_val_score(model,X,Y,cv = kfold)


# In[15]:


print(result.mean())


# In[16]:


from sklearn.model_selection import GridSearchCV


# In[18]:


n_neighbors = np.array(range(1,170))
param_grid = dict(n_neighbors=n_neighbors)
model=KNeighborsClassifier()
grid = GridSearchCV(estimator = model,param_grid = param_grid)
grid.fit(X,Y)


# In[19]:


print(grid.best_score_)
print(grid.best_params_)

