#!/usr/bin/env python
# coding: utf-8

# # Assignment NO.18
Forecast the CocaCola prices and Airlines Passengers data set. Prepare a document for each model explaining 
how many dummy variables you have created and RMSE value for each model. Finally which model you will use for 
Forecasting.
# In[3]:


from datetime import datetime
import numpy as np             #for numerical computations like log,exp,sqrt etc
import pandas as pd            #for reading & storing data, pre-processing
import matplotlib.pylab as plt #for visualization
#for making sure matplotlib plots are generated in Jupyter notebook itself
get_ipython().run_line_magic('matplotlib', 'inline')


# In[5]:


Air = pd.read_excel("C:/Users/Administrator/Downloads/Airlines+Data.xlsx")


# In[7]:


Air.head()


# In[8]:


Air.shape


# In[9]:


Air.info()


# In[10]:


# Visualizing the overall data in order to the components present in our data
plt.title("Line Plot", size = 15, weight = 'bold')
plt.ylabel("Passengers", size = 10, weight = 'bold')
plt.plot(Air['Passengers'])


# In[11]:


Air["month"] = Air.Month.dt.strftime("%b")


# In[12]:


Air


# In[13]:


data = pd.get_dummies(Air["month"])


# In[14]:


data


# In[15]:


Air1 = pd.concat([Air,data],axis=1)


# In[16]:


#find t_squared values and log values
Air1["t"] = np.arange(1,97)
Air1["t_squared"] = Air1["t"]*Air1["t"]
Air1.columns
Air1["log_passengers"] = np.log(Air1["Passengers"])


# In[17]:


Air1


# In[18]:


train= Air1.head(88)
test=Air1.tail(8)
Air1.Passengers.plot()


# In[19]:


indexedDataset = Air1.set_index(['Month'])
indexedDataset.head(5)


# In[20]:


rolmean = indexedDataset.rolling(window=12).mean() #window size 12 denotes 1 Year, giving rolling mean at yearly level
rolstd = indexedDataset.rolling(window=12).std()
print(rolmean,rolstd)


# In[21]:


Air1.Passengers.plot(label="org")
for i in range(2,10,2):
    Air1["Passengers"].rolling(i).mean().plot(label=str(i))
plt.legend(loc=3)


# In[22]:


import statsmodels.formula.api as smf


# In[23]:


#linear model
linear= smf.ols('Passengers~t',data=Air1).fit()
predlin=pd.Series(linear.predict(pd.DataFrame(test['t'])))
rmselin=np.sqrt((np.mean(np.array(test['Passengers'])-np.array(predlin))**2))
print("Root Mean Square Error : ",rmselin)


# In[24]:


#quadratic model
quad=smf.ols('Passengers~t+t_squared',data=Air1).fit()
predquad=pd.Series(quad.predict(pd.DataFrame(test[['t','t_squared']])))
rmsequad=np.sqrt(np.mean((np.array(test['Passengers'])-np.array(predquad))**2))
print("Root Mean Square Error : ",rmsequad)


# In[25]:


#exponential model
expo=smf.ols('log_passengers~t',data=Air1).fit()
predexp=pd.Series(expo.predict(pd.DataFrame(test['t'])))
predexp
rmseexpo=np.sqrt(np.mean((np.array(test['Passengers'])-np.array(np.exp(predexp)))**2))
print("Root Mean Square Error : ",rmseexpo)


# In[26]:


#additive seasonality
additive= smf.ols('Passengers~ Jan+Feb+Mar+Apr+May+Jun+Jul+Aug+Sep+Oct+Nov+Dec',data=Air1).fit()
predadd=pd.Series(additive.predict(pd.DataFrame(test[['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']])))
predadd
rmseadd=np.sqrt(np.mean((np.array(test['Passengers'])-np.array(predadd))**2))
print("Root Mean Square Error : ",rmseadd)


# In[27]:


#additive seasonality with linear trend
addlinear= smf.ols('Passengers~t+Jan+Feb+Mar+Apr+May+Jun+Jul+Aug+Sep+Oct+Nov+Dec',data=Air1).fit()
predaddlinear=pd.Series(addlinear.predict(pd.DataFrame(test[['t','Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']])))
predaddlinear


# In[28]:


rmseaddlinear=np.sqrt(np.mean((np.array(test['Passengers'])-np.array(predaddlinear))**2))
print("Root Mean Square Error : ",rmseaddlinear)


# In[29]:


#additive seasonality with quadratic trend
addquad=smf.ols('Passengers~t+t_squared+Jan+Feb+Mar+Apr+May+Jun+Jul+Aug+Sep+Oct+Nov+Dec',data=Air1).fit()
predaddquad=pd.Series(addquad.predict(pd.DataFrame(test[['t','t_squared','Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']])))
rmseaddquad=np.sqrt(np.mean((np.array(test['Passengers'])-np.array(predaddquad))**2))
print("Root Mean Square Error : ",rmseaddquad)


# In[30]:


#multiplicative seasonality
mulsea=smf.ols('log_passengers~Jan+Feb+Mar+Apr+May+Jun+Jul+Aug+Sep+Oct+Nov+Dec',data=Air1).fit()
predmul= pd.Series(mulsea.predict(pd.DataFrame(test[['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']])))
rmsemul= np.sqrt(np.mean((np.array(test['Passengers'])-np.array(np.exp(predmul)))**2))
print("Root Mean Square Error : ",rmsemul)


# In[31]:


#multiplicative seasonality with quadratic trend
mul_quad= smf.ols('log_passengers~t+t_squared+Jan+Feb+Mar+Apr+May+Jun+Jul+Aug+Sep+Oct+Nov+Dec',data=Air1).fit()
pred_mul_quad= pd.Series(mul_quad.predict(test[['t','t_squared','Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']]))
rmse_mul_quad=np.sqrt(np.mean((np.array(test['Passengers'])-np.array(np.exp(pred_mul_quad)))**2))
print("Root Mean Square Error : ",rmse_mul_quad)


# In[32]:


data={'Model':pd.Series(['rmse_mul_quad','rmseadd','rmseaddlinear','rmseaddquad','rmseexpo','rmselin','rmsemul','rmsequad']),'Values':pd.Series([rmse_mul_quad,rmseadd,rmseaddlinear,rmseaddquad,rmseexpo,rmselin,rmsemul,rmsequad])}


# In[33]:


Rmse=pd.DataFrame(data)
Rmse


# In[39]:


#import dataset for making model with rmes
data_Predict = pd.read_excel("C:/Users/Administrator/Downloads/Airlines+Data.xlsx")


# In[41]:


data_Predict


# In[35]:


#final model with least rmse value
Final_pred = smf.ols('log_passengers~t+t_squared+Jan+Feb+Mar+Apr+May+Jun+Jul+Aug+Sep+Oct+Nov+Dec',data=Air1).fit()


# In[36]:


pred_new  = pd.Series(Final_pred.predict(Air1))


# In[37]:


pred_new


# In[ ]:




