#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from sklearn .metrics import pairwise_distances
from scipy.spatial.distance import cosine
import warnings


# In[3]:


book=pd.read_csv("C:/Users/DELL/Desktop/ExcelR/code file/book.csv")


# In[4]:


book


# In[5]:


book.head()


# In[6]:


book_data = book.rename({'User.ID':'userid','Book.Title':'booktitle','Book.Rating':'bookrating'},axis = 1)


# In[7]:


book_data


# In[8]:


book_data.head()


# In[10]:


book1 = book_data.drop(['Unnamed: 0'], axis = 1)


# In[11]:


book1


# In[12]:


book1.head()


# In[13]:


len(book1['userid'].unique())


# In[14]:


array_user = book1['userid'].unique()


# In[15]:


array_user


# In[16]:


len(book1['booktitle'].unique())


# In[17]:


book_data1 = book1.pivot_table(index = 'userid',
                              columns = 'booktitle',
                              values = 'bookrating').reset_index(drop = True)


# In[18]:


book_data1


# In[19]:


book_data1.head()


# In[20]:


book_data1.index = book1.userid.unique()


# In[21]:


book_data1.fillna(0, inplace = True)


# In[23]:


book_data1


# In[24]:


book_data1.head()


# In[25]:


warnings.filterwarnings("ignore")


# In[27]:


user = 1 - pairwise_distances(book_data1.values,metric = 'cosine')


# In[28]:


user


# In[29]:


user_data = pd.DataFrame(user)


# In[30]:


user_data


# In[31]:


user_data.iloc[0:5,0:5]


# In[32]:


np.fill_diagonal(user,0)


# In[33]:


user_data.idxmax(axis = 1)


# In[34]:


book1[(book1['userid']==162107) |(book1['userid'] == 276726)]


# In[35]:


book1[(book1['userid']==276729) | (book1['userid'] == 276726)]


# In[36]:


user_1 = book1[book1['userid'] == 276729]


# In[37]:


user_2 = book1[book1['userid'] == 276726]


# In[38]:


pd.merge(user_1,user_2, on = 'booktitle', how = 'outer')


# In[ ]:




