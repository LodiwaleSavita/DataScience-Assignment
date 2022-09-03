#!/usr/bin/env python
# coding: utf-8

# In[20]:


import pandas as pd


# In[21]:


crime = pd.read_csv("C:/Users/DELL/Downloads/crime_data.csv")


# In[22]:


crime


# In[23]:


def norm_func(i):
    x = (i-i.min())/(i.max()-i.min())
    return(x)


# In[24]:


df_norm = norm_func(crime.iloc[:,1:])
df_norm


# In[25]:


import scipy.cluster.hierarchy as sch
dendrogram = sch.dendrogram(sch.linkage(df_norm,method='average'))


# In[26]:


from sklearn.cluster import AgglomerativeClustering
hc = AgglomerativeClustering(n_clusters=4,affinity = 'euclidean',linkage = 'complete')


# In[27]:


y_hc = hc.fit_predict(df_norm)
clusters=pd.DataFrame(y_hc,columns=['Clusters'])


# In[28]:


clusters


# In[29]:


crime['h_clusterid']=hc.labels_


# In[30]:


crime


# In[31]:


from sklearn.cluster import KMeans


# In[32]:


model = KMeans(n_clusters=5)
model.fit(df_norm)
model.labels_


# In[33]:


md=pd.Series(model.labels_)#converting numpy array into pandas series object
crime['Clust']=md # creating a new column and assigning it to new column
df_norm.head()


# In[34]:


crime.iloc[:,1:5].groupby(crime.Clust).mean()


# In[35]:


crime.head()


# In[37]:


from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler


# In[38]:


df = pd.read_csv("C:/Users/DELL/Downloads/crime_data.csv")


# In[39]:


df


# In[40]:


crime.info()


# In[41]:


df=crime.iloc[:,1:5]


# In[42]:


df


# In[43]:


array = df.values


# In[44]:


stscaler=StandardScaler().fit(array)
X=stscaler.transform(array)


# In[45]:


X


# In[46]:


dbscan = DBSCAN(eps=2,min_samples=5)
dbscan.fit(X)


# In[47]:


dbscan.labels_


# In[48]:


cl=pd.DataFrame(dbscan.labels_,columns=['cluster'])


# In[49]:


cl


# In[50]:


pd.concat([df,cl],axis=1)


# In[ ]:




