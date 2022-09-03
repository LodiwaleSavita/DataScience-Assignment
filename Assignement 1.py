#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# # Question7

# In[ ]:


Q=pd.read_csv("C:/Users/DELL/Downloads/Q7.csv")


# In[3]:


Q


# In[4]:


Q.mean()


# In[5]:


Q.median()


# In[6]:


Q.Points.mode()


# In[7]:


Q.Score.mode()


# In[8]:


Q.Weigh.mode()


# In[9]:


Q.var()


# In[10]:


Q.describe()


# In[14]:


fax=plt.subplots(figsize=(15,5))
plt.subplot(1,3,1)
plt.boxplot(Q.Points)
plt.title('Points')
plt.subplot(1,3,2)
plt.boxplot(Q.Score)
plt.title('Score')
plt.subplot(1,3,3)
plt.boxplot(Q.Weigh)
plt.title('weigh')
plt.show()


# # Question 9

# In[16]:


import pandas as pd


# In[17]:


data =pd.read_csv("C:/Users/DELL/Downloads/Q9_a.csv")


# In[18]:


data


# In[19]:


data.mean()


# In[20]:


data['Index'].mean()


# In[22]:


data['Index'].describe()


# In[23]:


data['speed'].skew()


# In[24]:


data['speed'].kurtosis()


# In[25]:


data['dist'].skew()


# In[27]:


data['dist'].kurtosis()


# # Question = 12

# In[28]:


import pandas as pd


# In[60]:


data2=pd.read_csv("C:/Users/DELL/Downloads/Q9_b.csv")


# In[61]:


data2


# In[32]:


data2.mean()


# In[33]:


data2['SP'].skew()


# In[34]:


data2['WT'].kurtosis()


# # Question12

# In[ ]:


import pandas as pd
import numpy as np
get_ipython().run_line_magic('matplotlib', 'inline')


# In[37]:


x=pd.Series([34,36,36,38,39,39,40,40,41.41,41,41,42,42,45,49,56])


# In[38]:


x.mean()


# In[39]:


x.median()


# In[40]:


x.median()


# In[41]:


x.var()


# In[42]:


x.std()


# # Question 2

# In[44]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import norm


# In[45]:


car=pd.read_csv("C:/Users/DELL/Downloads/Cars.csv")


# In[46]:


car


# In[47]:


sns.boxplot(car.MPG)


# In[48]:


1-stats.norm.cdf(38,car.MPG.mean(),car.MPG.std())


# In[50]:


stats.norm.cdf(40,car.MPG.mean(),car.MPG.std())


# In[51]:


stats.norm.cdf(0.50,car.MPG.mean(),car.MPG.std())-stats.norm.cdf(0.20,car.MPG.mean(),car.MPG.std())


# # Question 24

# In[52]:


from scipy import stats
from scipy.stats import norm


# In[53]:


t = (260-270)/(90/18**0.5)


# In[57]:


t


# In[55]:


p_value=1-stats.t.cdf(abs(0.4714),df=17)


# In[56]:


p_value

