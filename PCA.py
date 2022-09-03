#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.preprocessing import scale
import seaborn as sns


# In[4]:


W = pd.read_csv("C:/Users/DELL/Downloads/wine.csv")


# In[5]:


W


# In[6]:


print(W.describe)


# In[7]:


W.head()


# In[8]:


W['Type'].value_counts()


# In[9]:


W=W.iloc[:,1:]


# In[10]:


W


# In[11]:


W.shape


# In[12]:


W.info()


# In[13]:


W_ary=W.values


# In[14]:


W_ary


# In[15]:


W_norm=scale(W_ary)
W_norm


# In[16]:


pca=PCA()


# In[17]:


pca_values = pca.fit_transform(W_norm)


# In[19]:


pca_values


# In[20]:


pca.components_


# In[21]:


var = pca.explained_variance_ratio_


# In[22]:


var


# In[23]:


var = np.cumsum(np.round(var,decimals=4)*100)


# In[24]:


var


# In[25]:


var = np.cumsum(np.round(var,decimals=4)*100)


# In[26]:


var


# In[28]:


plt.plot(var,color ="red")


# # FINAL DATAFRAME

# In[33]:


final_df=pd.concat([W['Ash'],pd.DataFrame(pca_values[:,0:3],columns=['PC1','PC2','PC3'])],axis=1)


# In[34]:


final_df


# In[36]:


sns.scatterplot(data=final_df,x='PC1',y='PC2',hue='Ash');


# In[37]:


pca_values[:,0:1]


# In[38]:


x = pca_values[:,0:1]
y = pca_values[:,1:2]
plt.scatter(x,y);


# In[39]:


# Create Dendrogram
import scipy .cluster.hierarchy as sch
dendrogram =sch.dendrogram(sch.linkage(W_norm,'average'))


# In[42]:


from sklearn.cluster import AgglomerativeClustering
hc = AgglomerativeClustering(n_clusters=4,affinity = 'euclidean',linkage = 'complete')


# In[44]:


y_hc = hc.fit_predict(W_norm)
Clusters=pd.DataFrame(y_hc,columns=['Clusters'])


# In[45]:


Clusters


# In[46]:


W['h_clusterid']=hc.labels_


# In[47]:


from sklearn.cluster import KMeans


# In[48]:


model = KMeans(n_clusters=4)
model.fit(W_norm)


# In[49]:


model.labels_


# In[50]:


md=pd.Series(model.labels_) #Converting numpy array into pandas Series object
W['clust']=md #Creating a new column and assigning it to new column
W_norm


# In[51]:


W.iloc[:,1:15].groupby(W.clust).mean()


# In[52]:


W.head()


# In[53]:


wcss=[]
for i in range(1,8):
    kmeans = KMeans(n_clusters = i,random_state=2)
    kmeans.fit(W_norm)
    wcss.append(kmeans.inertia_)


# In[54]:


plt.plot(range(1,8),wcss)
plt.title('Elbow Graph')
plt.xlabel('Number of cluster')
plt.ylabel('WCSS')

