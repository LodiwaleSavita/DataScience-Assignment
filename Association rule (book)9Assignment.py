#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from mlxtend.frequent_patterns import apriori,association_rules
from mlxtend.preprocessing import TransactionEncoder


# In[9]:


books=pd.read_csv("C:/Users/DELL/Desktop/ExcelR/Savita/book (1).csv")


# In[ ]:


books


# In[10]:


books.info()


# # Apriori Algorithm

# In[13]:


# at min_support = 0.2 , support values will be more then 0.2 
frequent_itemsets_2 = apriori(books, min_support = 0.2,use_colnames=True)
frequent_itemsets_2


# In[15]:


rules2 = association_rules(frequent_itemsets_2,metric = "lift",min_threshold = 0.7)
rules2


# In[17]:


rules2.sort_values('lift',ascending = False)[0:20]


# In[18]:


rules2[rules2.lift>1]


# In[19]:


df1 = pd.DataFrame(data = frequent_itemsets_2)
df1
df1.duplicated()


# In[20]:


#visualization of obtained rule
rules2.plot(kind = 'bar',x ='support',y ='confidence',color = 'darkblue')
plt.title('Barplot')
plt.xlabel('support')
plt.ylabel('confidence')


# In[21]:


plt.scatter(rules2.support,rules2.confidence)
plt.xlabel('support')
plt.ylabel('confidence')
plt.show()


# In[23]:


# at min_support=0.17 , support values will be greater then 0.17
frequent_itemsets_17 = apriori (books,min_support = 0.17,use_colnames=True)
frequent_itemsets_17


# In[24]:


rules17 = association_rules(frequent_itemsets_17,metric = "lift",min_threshold=0.7)
rules17


# In[25]:


rules17.sort_values('lift',ascending = False)[0:20]


# In[28]:


rules17[rules17.lift>1]


# In[29]:


df17 = pd.DataFrame(data = frequent_itemsets_17)
df17
df17.duplicated()


# In[31]:


rules17.plot(kind='bar',x ='support',y = 'confidence',color = 'red')
plt.title('Barplot')
plt.xlabel('support')
plt.ylabel('confidence')
plt.show()


# In[32]:


plt.scatter(rules17.support,rules17.confidence)
plt.xlabel('support')
plt.ylabel('confidence')
plt.show()


# In[33]:


# at min_support  = 0.15 ,support values will be greater then 0.17
frequent_itemsets_15 = apriori(books,min_support = 0.15,use_colnames = True)
frequent_itemsets_15


# In[34]:


rules15 = association_rules(frequent_itemsets_15,metric = "lift",min_threshold =0.7)
rules15


# In[35]:


rules15.sort_values('lift',ascending = False)[0:20]


# In[39]:


df15 = pd.DataFrame(data = frequent_itemsets_15)
df15
df15.duplicated()


# In[41]:


rules15.plot(kind = 'bar', x = 'support', y = 'confidence',color = 'purple')
plt.title('Barplot')
plt.xlabel('support')
plt.ylabel('confidence')
plt.figure(figsize =(20,10))


# In[42]:


plt.scatter(rules15.support,rules15.confidence)
plt.xlabel('support')
plt.ylabel('confidence')
plt.show()


# In[43]:


x =[0.15,0.17,0.2]
y =[21,9,2]
plt.xlabel('Minimum Support')
plt.ylabel('Frequent Itemsets')
plt.title('Relation Between Min Support Value and Frequent Itemsets')


# In[45]:


x=[0.1,0.2,0.3]
y=[52,12,6]
plt.scatter(x,y)
plt.xlabel('Minimum Support')
plt.ylabel('Frequent Itemsets')
plt.title('Relation Between Min Support Value and Frequent Itemsets')


# In[ ]:




