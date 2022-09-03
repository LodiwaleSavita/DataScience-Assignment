#!/usr/bin/env python
# coding: utf-8

# In[13]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns
import statsmodels.formula.api as smf
from pandas.plotting import lag_plot
from statsmodels.graphics.tsaplots import plot_acf 


# In[14]:


data = pd.read_csv("C:/Users/Administrator/Desktop/ExcelR/CocaCola_Sales_Rawdata.csv",parse_dates=[0])


# In[15]:


data.head()


# In[16]:


data.Sales.plot()


# In[17]:


quarter=['Q1','Q2','Q3','Q4']
year=['86','87','88','89','90','91','92','93','94','95','96']
n=data['Quarter'][0]
n[0:2]
data['quarter']=0


# In[18]:


for i in range(42):
    n=data['Quarter'][i]
    data['quarter'][i]=n[0:2]


# In[19]:


data


# In[20]:


data1 = pd.get_dummies(data['quarter'])


# In[21]:


revised = pd.concat([data,data1],axis=1) 


# In[22]:


revised


# In[24]:


revised["t"] = np.arange(1,43)


# In[25]:


revised["t_squared"] = revised["t"]*revised["t"]
revised.columns
revised["log_Sales"] = np.log(revised["Sales"])


# In[26]:


revised


# In[27]:


import seaborn as sns
plt.figure(figsize=(25,5))
sns.lineplot(x="Quarter",y="Sales",data=revised)


# In[28]:


#creat heatmap
plt.figure(figsize=(12,8))
heatmap_y_quarter = pd.pivot_table(data=revised,values="Sales",index="Quarter",columns="quarter",aggfunc="mean", fill_value = 0)
sns.heatmap(heatmap_y_quarter,annot=True,fmt="g") #fmt is format of the grid values


# In[29]:


# Boxplot for ever
plt.figure(figsize=(8,6))
sns.boxplot(x="quarter",y="Sales",data=revised)


# In[30]:


import statsmodels.formula.api as smf
from pandas.plotting import lag_plot
from statsmodels.graphics.tsaplots import plot_acf 
# lag plot to know the relationship
lag_plot(revised['Sales'])
plt.title("Lag Plot", size = 15, weight = "bold")
plt.show()


# In[31]:


#AutoCorrelation Function plot #sales vs time
plot_acf(revised['Sales'], lags = 30, color = 'darkblue')               # lags = 30 means it will plot for k = 30 lags 
plt.xlabel("No of lags, k = 30", size = 10, weight = 'bold')
plt.ylabel("Autocorrelation (r2 value)", size = 20, weight = 'bold')
plt.show()


# In[32]:


#plot lineplot with quarter and sales
plt.figure(figsize=(12,3))
sns.lineplot(x="quarter",y="Sales",data=revised)


# In[33]:


# Splitting Data into train and test

train =revised.head(37)
test  =revised.tail(4)


# In[34]:


test


# In[35]:


#linear model

linear = smf.ols('Sales~t',data=train).fit()
pred_linear =  pd.Series(linear.predict(pd.DataFrame(test['t'])))
rmse_linear = np.sqrt(np.mean((np.array(test['Sales'])-np.array(pred_linear))**2))
print("Root Mean Square Error : ",rmse_linear)


# In[36]:


#quadratic model

Quad = smf.ols('Sales~t+t_squared',data=train).fit() #quadratic model
pred_Quad = pd.Series(Quad.predict(test[["t","t_squared"]]))
rmse_Quad = np.sqrt(np.mean((np.array(test['Sales'])-np.array(pred_Quad))**2))
print("Root Mean Square Error : ",rmse_Quad)


# In[37]:


#exponential model

Exp = smf.ols('log_Sales~t',data=train).fit() #exponential model
pred_Exp = pd.Series(Exp.predict(pd.DataFrame(test['t'])))
rmse_Exp = np.sqrt(np.mean((np.array(test['Sales'])-np.array(np.exp(pred_Exp)))**2))
print("Root Mean Square Error : ",rmse_Exp)


# In[38]:


#additive seasonality

add_sea = smf.ols('Sales~Q1+Q2+Q3+Q4',data=train).fit() #additive seasonality model
pred_add_sea = pd.Series(add_sea.predict(test[['Q1','Q2','Q3','Q4']]))
rmse_add_sea = np.sqrt(np.mean((np.array(test['Sales'])-np.array(pred_add_sea))**2))
print("Root Mean Square Error : ",rmse_add_sea)


# In[39]:


#additive seasonality with linear treand

add_sea_quad = smf.ols('Sales~t+t_squared+Q1+Q2+Q3+Q4',data=train).fit() #additive seasonality qudratic model
pred_add_sea_quad = pd.Series(add_sea_quad.predict(test[['t','t_squared','Q1','Q2','Q3','Q4']]))
rmse_add_sea_quad = np.sqrt(np.mean((np.array(test['Sales'])-np.array(pred_add_sea_quad))**2))
print("Root Mean Square Error : ",rmse_add_sea_quad)


# In[40]:


#additive seasonality with quadratic trend

add_sea_quad = smf.ols('Sales~t+t_squared+Q1+Q2+Q3+Q4',data=train).fit() #additive seasonality qudratic model
pred_add_sea_quad = pd.Series(add_sea_quad.predict(test[['t','t_squared','Q1','Q2','Q3','Q4']]))
rmse_add_sea_quad = np.sqrt(np.mean((np.array(test['Sales'])-np.array(pred_add_sea_quad))**2))
print("Root Mean Square Error : ",rmse_add_sea_quad)


# In[41]:


#multiplicative seasonality

Mul_sea = smf.ols('log_Sales~Q1+Q2+Q3+Q4',data = train).fit() #multiplicative seasonality model
pred_Mult_sea = pd.Series(Mul_sea.predict(test))
rmse_Mult_sea = np.sqrt(np.mean((np.array(test['Sales'])-np.array(np.exp(pred_Mult_sea)))**2))
print("Root Mean Square Error : ",rmse_Mult_sea)


# In[42]:


#multiplicative additive seasonality

Mul_Add_sea = smf.ols('log_Sales~t+Q1+Q2+Q3+Q4',data = train).fit() #multiplicative additive seasonality
pred_Mult_add_sea = pd.Series(Mul_Add_sea.predict(test))
rmse_Mult_add_sea = np.sqrt(np.mean((np.array(test['Sales'])-np.array(np.exp(pred_Mult_add_sea)))**2))
print("Root Mean Square Error : ",rmse_Mult_add_sea)


# In[43]:


#tabuling rmes value
Final_data = {"MODEL":pd.Series(["rmse_linear","rmse_Exp","rmse_Quad","rmse_add_sea","rmse_add_sea_quad","rmse_Mult_sea","rmse_Mult_add_sea"]),"RMSE_Values":pd.Series([rmse_linear,rmse_Exp,rmse_Quad,rmse_add_sea,rmse_add_sea_quad,rmse_Mult_sea,rmse_Mult_add_sea])}
Final_result = pd.DataFrame(Final_data) #data frame of final result
Final_result.sort_values(['RMSE_Values'])


# In[45]:


#import dataset for making model with rmes
data_Predict = pd.read_csv("C:/Users/Administrator/Desktop/ExcelR/CocaCola_Sales_Rawdata.csv")


# In[46]:


#final model with least rmse value
Final_pred = smf.ols('log_Sales~t+Q1+Q2+Q3+Q4',data=revised).fit()


# In[47]:


pred_new  = pd.Series(Final_pred.predict(revised))


# In[48]:


pred_new


# In[ ]:




