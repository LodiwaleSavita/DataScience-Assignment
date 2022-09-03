#!/usr/bin/env python
# coding: utf-8

# In[6]:


import pandas as pd
import numpy as np


# In[7]:


data = pd.read_csv("C:/Users/DELL/Downloads/delivery_time.csv")


# In[8]:


data


# In[9]:


data.head()


# In[10]:


data.info()


# In[11]:


data.corr()


# In[12]:


import seaborn as sns
sns.distplot(data['Sorting Time'])


# In[13]:


sns.distplot(data['Delivery Time'])


# In[11]:


data = data.rename({'Delivery Time':'delivery_time','Sorting Time':'sorting_time'},axis=1)


# In[14]:


data


# In[15]:


import statsmodels.formula.api as smf


# In[16]:


sns.regplot(x = data['sorting_time'],y = data['delivery_time'],data = data)
model = smf.ols("delivery_time ~ sorting_time",data = data).fit()
model


# In[2]:


model


# In[19]:


model.params


# In[20]:


print(model.tvalues,'\n',model.pvalues)


# In[1]:


(model.rsquared,model.rsquared_adj)


# In[22]:


y = (6.582734)+(1.649020)*5


# In[23]:


y


# In[24]:


newtime = pd.Series([5,8])


# In[25]:


pred =pd.DataFrame(newtime,columns=['sorting_time'])
pred


# In[27]:


model.predict(pred)


# # log Transformation

# In[31]:


x_log=np.log(data['sorting_time'])
y_log=np.log(data['delivery_time'])


# In[32]:


model = smf.ols("y_log~ x_log",data = data).fit()


# In[33]:


model


# In[34]:


model.params


# In[35]:


print(model.pvalues,'\n',model.tvalues)


# In[36]:


model.rsquared,model.rsquared_adj


# In[37]:


y_log = (1.741987)+(0.597522)*5


# In[38]:


y_log


# In[39]:


newtime = pd.Series([5,8])


# In[40]:


pred = pd.DataFrame(newtime,columns=['x_log'])


# In[41]:


pred


# In[42]:


model.predict(pred)


# In[43]:


data


# # Improving Model Using Squareroot Transformation

# In[45]:


data.insert(len(data.columns),'a_sqrt',
           np.sqrt(data.iloc[:,0]))


# In[46]:


data


# In[ ]:


model = smf.ols("delivery_time ~ a_sqrt",data = data").fit()


# In[47]:


model


# In[48]:


model.params


# In[49]:


print(model.pvalues,'\n',model.tvalues)


# In[51]:


(model.rsquared,model.rsquared_adj)


# In[53]:


y_quad = (-3.930699)+(3.977225)*5


# In[54]:


y_quad


# In[55]:


newtime=pd.Series([5,8])


# In[57]:


pred=pd.DataFrame(newtime,columns = ['a_sqrt'])


# # Improving model with SquareTransformation

# In[69]:


data['Squar_del_time']=data.apply(lambda row:row.delivery_time**2,axis =1)


# In[70]:


data


# In[71]:


model = smf.ols('Squar_del_time ~ sorting_time',data = data).fit()


# In[72]:


model


# In[65]:


model.params


# In[66]:


print(model.pvalues,'\n',model.tvalues)


# In[67]:


(model.rsquared,model.rsquared_adj)


# # Improvement Model With Reciprocol Transformation
# 

# In[73]:


reciprocal_del_time=1/data["delivery_time"]


# In[74]:


reciprocal_del_time


# In[75]:


model = smf.ols('reciprocal_del_time~sorting_time',data = data).fit()


# In[76]:


model


# In[77]:


model.params


# In[78]:


print(model.pvalues,'\n',model.tvalues)


# In[79]:


(model.rsquared,model.rsquared_adj)


# # Improving model using Box - cox Transformation

# In[80]:


from scipy.stats import boxcox
bcx_target,lam = boxcox(data["delivery_time"])


# In[81]:


model = smf.ols('bcx_target~sorting_time',data = data).fit()


# In[82]:


model.params


# In[83]:


print(model.pvalues,'\n',model.tvalues)


# In[85]:


(model.rsquared,model.rsquared_adj)


# # Improving model using yeo-johson transformation

# In[89]:


from scipy.stats import yeojohnson
yf_target,lam = yeojohnson(data["delivery_time"])


# In[90]:


model = smf.ols('yf_target~sorting_time',data=data).fit()


# In[91]:


model.params


# In[92]:


print(model.pvalues,'\n',model.tvalues)


# In[93]:


(model.rsquared,model.rsquared_adj)


# # Model.rsquared,model.rsquared_adj
# 

# # The Reciprocol transformation is best transformation for this model

# # Statement2

# In[96]:


salary = pd.read_csv("C:/Users/DELL/Downloads/Salary_Data.csv")


# In[97]:


salary


# In[98]:


salary.corr()


# In[99]:


sns.distplot(salary['YearsExperience'])


# In[100]:


sns.displot(salary['Salary'])


# In[104]:


salary = salary.rename({'YearsExperience':'year','Salary':'income'},axis=1)


# In[105]:


salary


# In[106]:


sns.regplot(x ='year',y ='income',data = salary)


# In[110]:


model = smf.ols ("income ~ year",data = salary).fit()


# In[111]:


model


# In[112]:


model.params


# In[113]:


print(model.tvalues,'\n',model.pvalues)


# In[114]:


(model.rsquared,model.rsquared_adj)


# In[115]:


newsalary = pd.Series([200,300])


# In[116]:


data_pred = pd.DataFrame(newsalary,columns = ['year'])


# In[117]:


data_pred


# In[118]:


model.predict(data_pred)


# # Improving Model Using logarithm

# In[119]:


salary1 = np.log(salary)


# In[120]:


salary1


# In[121]:


sns.regplot(x ='year',y='income',data = salary1)


# In[122]:


model =smf.ols("income ~ year",data = salary1).fit()


# In[123]:


print(model.pvalues,'\n',model.tvalues)


# In[124]:


(model.rsquared)


# # Improving Model Using Squarroot Transformation

# In[128]:


salary.insert(len(salary.columns),'A_sqrt',
             np.sqrt(salary.iloc[:,0]))


# In[129]:


salary


# In[131]:


model =smf.ols('income~A_sqrt',data = salary).fit()


# In[132]:


model


# In[133]:


model.params


# In[134]:


print(model.tvalues,'\n',model.pvalues)


# In[135]:


(model.rsquared,model.rsquared_adj)


# # Improving Model with Square Transformation

# In[149]:


salary['Squar_income']=salary.apply(lambda row:row.income**2,axis = 1)


# In[150]:


salary


# In[151]:


model=smf.ols('Squar_income~year',data=salary).fit()


# In[152]:


model


# In[153]:


model.params


# In[155]:


print(model.pvalues,'\n',model.tvalues)


# In[157]:


(model.rsquared,model.rsquared_adj)


# # Improvement Model using box - cox transformation 

# In[158]:


from scipy.stats import boxcox
bcx_target,lam = boxcox(salary["income"])


# In[159]:


model = smf.ols('bcx_target ~ year',data = salary).fit()


# In[160]:


model


# In[161]:


print(model.pvalues,'\n',model.tvalues)


# In[162]:


(model.rsquared,model.rsquared_adj)


# # improving model using yeo-johnson transformation

# In[163]:


from scipy.stats import yeojohnson
yf_target,lam = yeojohnson(salary["income"])


# In[168]:


model = smf.ols('yf_target~year',data=salary).fit()


# In[165]:


model.params


# In[166]:


print(model.pvalues,'\n',model.tvalues)


# In[167]:


(model.rsquared,model.rsquared_adj)

