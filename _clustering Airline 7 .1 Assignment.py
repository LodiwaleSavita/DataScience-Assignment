#!/usr/bin/env python
# coding: utf-8

# In[32]:


#import hierarchical clustering libraries
import scipy.cluster.hierarchy as sch
from sklearn.cluster import AgglomerativeClustering
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sn


# In[33]:


Airline=pd.read_csv("C:/Users/DELL/Downloads/EastWestAirlines.csv")


# In[34]:


Airline


# In[35]:


# Normalization function
def norm_func(i):
    x = (i-i.min())/(i.max()-i.min())
    return(x)


# In[36]:


df_norm = norm_func(Airline.iloc[:,1:])
df_norm


# In[37]:


#Normalized data frame (considering the numerical part of data)
df_norm = norm_func(Airline.iloc[:,1:])


# In[38]:


# create dendrogram
dendrogram = sch.dendrogram(sch.linkage(df_norm,method='ward'))


# In[39]:


hc = AgglomerativeClustering(n_clusters=4, affinity = 'euclidean',linkage = 'complete')


# In[40]:


y_hc = hc.fit_predict(df_norm)
Clusters=pd.DataFrame(y_hc,columns=['Clusters'])


# In[41]:


Clusters


# In[42]:


Airline['h_clusterid']=hc.labels_


# In[43]:


Airline


# In[44]:


from sklearn.cluster import KMeans


# In[45]:


model = KMeans(n_clusters=5)
model.fit(df_norm)


# In[46]:


model.labels_


# In[47]:


md=pd.Series(model.labels_)#Converting numpy array into pandas series object 
Airline['clust']=md # creating a new column and assigning it to new column
df_norm.head()


# In[48]:


Airline.iloc[:,1:12].groupby(Airline.clust).mean()


# In[57]:


from sklearn.cluster import DBSCAN
from sklearn.preprocessing import  StandardScaler


# In[58]:


Airline = pd.read_csv("C:/Users/DELL/Downloads/EastWestAirlines.csv")


# In[59]:


Airline


# In[60]:


Airline.info()


# In[61]:


df = Airline.iloc[:,1:12]


# In[62]:


df


# In[63]:


array = df.values


# In[65]:


stscaler=StandardScaler().fit(array)
X = stscaler.transform(array)


# In[66]:


X


# In[68]:


dbscan = DBSCAN(eps=2,min_samples=5)
dbscan.fit(X)


# In[69]:


dbscan.labels_


# In[70]:


cl=pd.DataFrame(dbscan.labels_,columns=['cluster'])


# In[71]:


cl


# In[72]:


pd.concat([df,cl],axis=1)


# In[ ]:




