#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import norm


# In[2]:


data=pd.read_csv("C:/Users/DELL/Downloads/Cutlets.csv")


# In[3]:


data


# In[4]:


UnitA=data.iloc[:,0]


# In[5]:


UnitA


# In[6]:


UnitB=data.iloc[:,1]


# In[7]:


UnitB


# In[8]:


#probability using t-test
p_value=stats.ttest_ind(UnitA,UnitB)


# In[9]:


p_value


# In[10]:


p_value[1]


# In[11]:


#Question 2


# In[42]:


import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import norm


# In[43]:


data=pd.read_csv("C:/Users/DELL/Downloads/LabTAT.csv")


# In[44]:


data


# In[14]:


p_value=stats.f_oneway(data.iloc[:,0],data.iloc[:,1],data.iloc[:,2],data.iloc[:,3])


# In[15]:


p_value


# In[16]:


p_value[1]


# In[17]:


# Question3


# In[45]:


import scipy.stats as sp
alpha=0.05
Male=[50,142,131,70]
Female=[435,1523,1356,750]
Sales=[Male,Female]
print(Sales)


# In[46]:


chitStats=sp.chi2_contingency(Sales)
print('Test t=%f p-value=%f'%(chitStats[0],chitStats[1]))
print('Interpret by p-Value')
if chitStats[1] < 0.05:
    print('we reject null hypothesis')
else:
    print('we accept null hypothesis')


# In[47]:


# Question 4


# In[1]:


import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import norm
from scipy.stats import chi2_contingency
from scipy.stats import chi2


# In[2]:


data=pd.read_csv("C:/Users/DELL/Downloads/Costomer+OrderForm.csv")


# In[3]:


data


# In[5]:


print(data['Phillippines'].value_counts(),data['Indonesia'].value_counts(),data['Malta'].value_counts(),data['India'].value_counts())


# In[6]:


observed=([[271,267,280],[29,33,31,20]])


# In[7]:


stat,p,dof,expected = chi2_contingency([[271,267,269,280],[29,33,31,20]])


# In[8]:


stat


# In[9]:


p


# In[10]:


print('dof=%d' % dof)
print(expected)


# In[11]:


alpha=0.05
prob=1-alpha
critical = chi2.ppf(prob ,dof)
print('probability=%.3f, critical=%.3f, stat=%.3f' % (prob,critical,stat))
if abs(stat) >=critical:
   print('Dependent (rejectH0),variables are related')
else:
   print('Independent (fail to reject H0),variables are not related')


# In[ ]:





# In[ ]:




