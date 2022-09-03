#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd
import seaborn as sns
import statsmodels.formula.api as smf


# In[4]:


t_c=pd.read_csv("C:/Users/DELL/Downloads/ToyotaCorolla.csv")


# In[5]:


t_c


# In[6]:


t_c.info()


# In[8]:


t_c1=t_c.iloc[:,[2,3,6,8,12,13,15,16,17]]


# In[9]:


t_c1


# In[10]:


t_c1.corr()


# In[11]:


t_c.isna().sum


# In[12]:


sns.set_style(style ='darkgrid')


# In[13]:


sns.pairplot(t_c1)


# # Model Building

# In[14]:


model = smf.ols('Price~Age_08_04+KM+HP+cc+Gears+Doors+Quarterly_Tax+Weight',data=t_c1).fit()


# In[15]:


model


# In[16]:


model.params


# In[17]:


print(model.tvalues,'\n',model.pvalues)


# In[18]:


(model.rsquared,model.rsquared_adj)


# In[19]:


slr_cc=smf.ols('Price~cc',data=t_c1).fit()


# In[20]:


slr_cc.params


# In[21]:


(slr_cc.pvalues,'\n,slr_cc.tvalues')


# In[22]:


(slr_cc.rsquared,slr_cc.rsquared_adj)


# In[23]:


slr_d=smf.ols('Price~Doors',data=t_c1).fit()


# In[24]:


slr_d.params


# In[25]:


print(slr_d.pvalues,'\n',slr_d.tvalues)


# In[26]:


(slr_d.rsquared,slr_d.rsquared_adj)


# In[27]:


(slr_cc.pvalues,slr_cc.tvalues)


# In[28]:


mlr_cc=smf.ols('Price~cc+Doors',data=t_c1).fit()


# In[29]:


mlr_cc.tvalues,mlr_cc.pvalues


# # Model Validation

# # Calculation VIF

# In[30]:


rsq_age=smf.ols('Age_08_04~KM+HP+cc+Gears+Doors+Quarterly_Tax+Weight',data =t_c1).fit().rsquared


# In[31]:


vif_age=1/(1-rsq_age)


# In[32]:


vif_age


# In[33]:


rsq_KM=smf.ols('KM~Age_08_04+HP+cc+Gears+Doors+Quarterly_Tax+Weight',data=t_c1).fit().rsquared


# In[34]:


vif_KM = 1/(1-rsq_KM)


# In[35]:


vif_KM


# In[36]:


HP = smf.ols('HP~KM+Age_08_04+cc+Doors+Gears+Quarterly_Tax+Weight',data=t_c1).fit().rsquared


# In[37]:


vif_HP=1/(1-HP)


# In[38]:


vif_HP


# In[39]:


cc=smf.ols('cc~Age_08_04+KM+HP+Doors+Gears+Quarterly_Tax+Weight',data=t_c1).fit().rsquared


# In[40]:


vif_cc =1/(1-cc)


# In[41]:


vif_cc


# In[42]:


doors=smf.ols('Doors~Age_08_04+KM+HP+cc+Gears+Quarterly_Tax+Weight',data=t_c1).fit().rsquared


# In[44]:


vif_doors=1/(1-doors)


# In[45]:


vif_doors


# In[46]:


Quarterly_Tax=smf.ols('Quarterly_Tax~Age_08_04+Gears+cc+KM+HP+Doors+Weight',data=t_c1).fit().rsquared


# In[47]:


vif_Quarterly_Tax = 1/(1-Quarterly_Tax)


# In[48]:


vif_Quarterly_Tax


# In[49]:


Weight = smf.ols('Weight~Age_08_04+Quarterly_Tax+cc+Gears+KM+HP+Doors',data=t_c1).fit().rsquared


# In[50]:


vif_Weight = 1/(1-Weight)


# In[51]:


Gears = smf.ols('Gears~Age_08_04+Quarterly_Tax+cc+KM+HP+Doors+Weight',data = t_c1).fit().rsquared


# In[52]:


vif_Gears=1/(1-Gears)


# In[53]:


vif_Gears


# In[54]:


dataframe={'variables':['Age_08_04','KM','HP','Gears','Doors','Quarterly_Tax','cc','Weight'],'vif':
         [vif_age,vif_KM,vif_HP,vif_cc,vif_doors,vif_Quarterly_Tax,vif_Weight,vif_Gears]}


# In[55]:


dataframe


# In[56]:


vif_df = pd.DataFrame(dataframe)


# In[57]:


vif_df


# In[58]:


import statsmodels.api as sm


# In[59]:


sm.qqplot(model.resid,line='q')


# In[60]:


import numpy as np


# In[61]:


def standardized(values):
    return (values-values.mean())/values.std()


# In[62]:


import matplotlib.pyplot as plt


# In[63]:


plt.scatter(standardized(model.fittedvalues),standardized(model.resid))


# # Residual Vs Regressor

# In[64]:


import statsmodels.api as sm


# In[65]:


fig =plt.figure(figsize=(15,8))
fig =sm.graphics.plot_regress_exog(model,"Age_08_04",fig=fig)
plt.show()


# In[66]:


import statsmodels.api as sm


# In[67]:


fig = plt.figure(figsize=(15,8))
fig = sm.graphics.plot_regress_exog(model,"HP",fig=fig)
plt.show()


# In[68]:


import statsmodels.api as sm


# In[69]:


fig = plt.figure(figsize=(15,8))
fig = sm.graphics.plot_regress_exog(model,"cc",fig=fig)
plt.show()


# In[70]:


import statsmodels.api as sm


# In[71]:


fig = plt.figure(figsize=(15,8))
fig = sm.graphics.plot_regress_exog(model,"Gears",fig=fig)
plt.show()


# In[72]:


import statsmodels.api as sm


# In[73]:


fig = plt.figure(figsize=(15,8))
fig = sm.graphics.plot_regress_exog(model,"Doors",fig=fig)
plt.show()


# In[74]:


import statsmodels.api as sm


# In[75]:


fig = plt.figure(figsize=(15,8))
fig = sm.graphics.plot_regress_exog(model,"Quarterly_Tax",fig=fig)
plt.show()


# In[76]:


import statsmodels.api as sm


# In[77]:


fig = plt.figure(figsize=(15,8))
fig = sm.graphics.plot_regress_exog(model,"Weight",fig=fig)
plt.show()


# #  cooks distance
# 

# In[78]:


model_influence = model.get_influence()
(c,_) = model_influence.cooks_distance


# In[79]:


fig = plt.subplots(figsize=(20, 7))
plt.stem(np.arange(len(t_c1)), np.round(c, 3))
plt.xlabel('Row index')
plt.ylabel('Cooks Distance')
plt.show()


# In[80]:


(np.argmax(c),np.max(c))


# # high influence point

# In[81]:


from statsmodels.graphics.regressionplots import influence_plot


# In[82]:


influence_plot(model)
plt.show()


# In[83]:


k = t_c1.shape[1]
n = t_c1.shape[0]
leverage_cutoff = 3*((k + 1)/n)


# In[86]:


# from above graph data pont 46 and 48 are influencer
t_c1[t_c1.index.isin([80])]


# In[88]:


t_c1.head()


# In[89]:


#improving the model

t_c2=pd.read_csv("C:/Users/DELL/Downloads/ToyotaCorolla.csv")


# In[90]:


t_c2


# In[91]:


#deleting influencer rows
t_c3=t_c2.drop(t_c2.index[[80]],axis=0).reset_index()


# In[92]:


t_c3


# In[93]:


t_c3=t_c3.drop(['index'],axis=1)


# In[94]:


t_c3


# In[95]:


final_mlr =smf.ols('Price~Age_08_04+KM+HP+cc+Gears+ Doors+Quarterly_Tax+Weight',data=t_c1).fit()


# In[96]:


(final_mlr.rsquared,final_mlr.aic)


# #  Model predication

# In[97]:


new_data=pd.DataFrame({"Age_08_04":30,"KM":20000,"HP":92,"cc":300,"Doors":4,"Gears":4,"Quarterly_Tax":200,"Weight":1000},index=[0])
new_data


# In[98]:


final_mlr.predict(new_data)


# In[99]:


pred_y=final_mlr.predict(t_c1)
pred_y


# In[ ]:




