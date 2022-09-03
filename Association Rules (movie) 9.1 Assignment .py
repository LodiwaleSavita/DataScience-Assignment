#!/usr/bin/env python
# coding: utf-8

# In[1]:


pip install mlxtend


# In[2]:


import pandas as pd 
import numpy as np 
import seaborn as sns
import matplotlib.pyplot as plt
from mlxtend.frequent_patterns import apriori,association_rules
from mlxtend.preprocessing import TransactionEncoder


# In[3]:


movies=pd.read_csv("C:/Users/DELL/Downloads/my_movies.csv")


# In[4]:


movies


# In[5]:


movies.info()


# In[6]:


movies1=movies.drop(['V1','V2','V3','V4','V5'],axis = 1)
movies1


# # Apriori Algorithm

# In[7]:


# with min_support of 0.1
frequent_itemsets1=apriori(movies1,min_support=0.1,use_colnames=True)
frequent_itemsets1


# In[8]:


rules1 = association_rules(frequent_itemsets1,metric="lift",min_threshold=0.7)
rules1


# In[9]:


rules1.sort_values('lift',ascending = False)[0:20]


# In[10]:


rules1[rules1.lift>1]


# In[11]:


df1=pd.DataFrame(data=frequent_itemsets1)
df1
df1.duplicated()


# In[12]:


#visualization of obtained rule
rules1.plot(kind='bar',x='support',y='confidence',color='green')
plt.title('Barplot')
plt.xlabel('support')
plt.ylabel('confidence')


# In[13]:


# plt.figure(figsize=(30,10))
#visualization of obtained rule
plt.scatter(rules1.support,rules1.confidence)
plt.xlabel('support')
plt.ylabel('confidence')
plt.show()


# In[14]:


#with min_support of 0.2
frequent_itemsets2=apriori(movies1,min_support=0.2,use_colnames=True)
frequent_itemsets2


# In[15]:


rules2 = association_rules(frequent_itemsets2,metric = "lift",min_threshold = 0.7)
rules2


# In[16]:


rules2.sort_values('lift',ascending = False)[0:20]


# In[17]:


rules2[rules2.lift>1]


# In[18]:


df2=pd.DataFrame(data=frequent_itemsets2)
df2
df2.duplicated()


# In[19]:


#visualization of obtained rule
rules2.plot(kind='bar',x='support',y='confidence',color='black')
plt.title('Barplot')
plt.xlabel('support')
plt.ylabel('confidence')
#plt.figure(figsize=(30,10))


# In[20]:


#visualization of obtained rule
plt.scatter(rules2.support,rules2.confidence)
plt.xlabel('support')
plt.ylabel('confidence')
plt.show()


# In[22]:


#with min_support of 0.3
frequent_itemsets3=apriori(movies1,min_support=0.3,use_colnames=True)
frequent_itemsets3


# In[23]:


rules3 = association_rules(frequent_itemsets3,metric = "lift",min_threshold = 0.7)
rules3


# In[24]:


rules3.sort_values('lift',ascending = False)[0:20]


# In[25]:


rules3


# In[26]:


rules3[rules3.lift>1]


# In[29]:


df3=pd.DataFrame(data=frequent_itemsets3)
df3
df3.duplicated()


# In[31]:


#Visualization of obtained rule
rules3.plot(kind = 'bar',x='support',y='confidence',color='green')
plt.title('Barplot')
plt.xlabel('support')
plt.ylabel('confidence')
#plt.figure(fifsize = (30,10))


# In[32]:


#visualizatiom of obtained rule
plt.scatter(rules3.support,rules3.confidence)
plt.xlabel('support')
plt.ylabel('confidence')
plt.show()


# In[33]:


x = [0.1,0.2,0.3]
y = [52,12,6]
plt.scatter(x,y)
plt.xlabel('Minimum Support')
plt.ylabel('Frequence Itemsets')
plt.title ('Relation Between Min Support Value and Frequence Itemsets')

