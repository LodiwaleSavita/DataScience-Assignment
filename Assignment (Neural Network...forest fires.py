#!/usr/bin/env python
# coding: utf-8

# ### FOREST FIRES DATASET

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import numpy as np
from keras.models import Sequential
from keras.layers import Dense


# In[3]:


forest = pd.read_csv("C:/Users/Administrator/Downloads/forestfires.csv")


# In[4]:


forest.head()


# # EDA

# In[5]:


forest.shape


# In[6]:


forest.head()


# In[7]:


forest.info()


# In[8]:


forest_1=forest[~forest.duplicated()]


# In[9]:


forest_1.reset_index(inplace=True)


# In[10]:


forest_1


# In[11]:


forest_1=forest_1.drop('index',axis=1)
forest_1


# In[12]:


plt.figure(figsize=(12,5))
forest_1.size_category.value_counts().plot.bar()


# In[15]:


plt.figure(figsize=(12,5))
forest_1.month.value_counts().plot.bar()


# In[18]:


# PLotting Month Vs. temp plot



plt.rcParams['figure.figsize'] = [20, 10]
sns.set(style = "darkgrid", font_scale = 1.3)
month_temp =sns.barplot(x = 'month', y = 'temp' ,data = forest_1,
                        order = ['jan','feb','mar','apr','may','jun','jul','aug','sep','oct','nov','dec'],palette = 'winter');
month_temp.set(title = "Month Vs Temp Barplot", xlabel = "Months", ylabel = "Temperature");


# # Preprocessing

# In[19]:


forest_2= forest_1.iloc[:,2:30]


# In[20]:


forest_2


# In[21]:


scaler = StandardScaler()


# In[22]:


forest_norm=scaler.fit_transform(forest_2)


# In[23]:


# number of columns are more need to use PCA 


# In[24]:


pca=PCA(n_components=28)
pca_values=pca.fit_transform(forest_norm)
pca_values


# In[25]:


variance = pca.explained_variance_ratio_
variance


# In[26]:


variance_1 = np.cumsum(np.round(variance,decimals = 4)*100)
variance_1


# In[27]:


# Variance graph of PCA
plt.figure(figsize=(15,6))
plt.plot(variance_1,color='red',marker='p')


# In[ ]:


# slecting first 25 pca out of 28


# In[29]:


final=pd.concat([pd.DataFrame(pca_values[:,:25]),forest_1[['size_category']]],axis=1)
final.size_category.replace(('large','small'),(1,0),inplace=True)


# In[30]:


final


# In[31]:


# Spliting data
array = final.values
x = array[:,0:25]
y= array[:,25]


# # MODEL BUILDING

# In[32]:


model = Sequential()
model.add(Dense(12, input_dim=25, activation='relu'))
model.add(Dense(8,activation='relu'))
model.add(Dense(1,activation='sigmoid'))


# In[34]:


#compile the model
model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
# Fit the model
model.fit(x, y, validation_split=0.3, epochs=200, batch_size=10)


# In[35]:


# Accuracy
scores = model.evaluate(x, y)
print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))


# In[ ]:




