#!/usr/bin/env python
# coding: utf-8

# # GAS TURBINES DATASET

# In[2]:


import pandas as pd
import pathlib
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Dense


# In[3]:


from __future__ import absolute_import, division, print_function


# In[4]:


gas=pd.read_csv("C:/Users/Administrator/Downloads/gas_turbines.csv")


# # EDA

# In[6]:


gas.shape


# In[7]:


gas.head()


# In[8]:


gas.tail()


# In[9]:


gas.info()


# In[10]:


gas[gas.duplicated()]


# In[11]:


# correlation with TEY

data2 = gas.copy()

correlations = data2.corrwith(gas["TEY"])
correlations = correlations[correlations!=1]
positive_correlations = correlations[correlations >0].sort_values(ascending = False)
negative_correlations =correlations[correlations<0].sort_values(ascending = False)

correlations.plot.bar(
        figsize = (18, 10), 
        fontsize = 15, 
        color = 'b',
        rot = 90, grid = True)
plt.title('Correlation with Turbine energy yield \n',
horizontalalignment="center", fontstyle = "normal", 
fontsize = "22", fontfamily = "sans-serif")


# # Preprocessing

# In[12]:


#Split the data into train and test
train_data=gas.sample(frac=0.8,random_state=0)
test_data=gas.drop(train_data.index)


# In[13]:


# Look overall statistics
train_stats=train_data.describe()
train_stats.pop('TEY')
train_stats=train_stats.transpose()
train_stats


# In[14]:


# Split features from labels
train_labels=train_data.pop('TEY')
test_labels=test_data.pop('TEY')


# In[15]:


# Normalize the data
def norm(x):
    return (x-train_stats['mean']) / train_stats['std']
normed_train_data=norm(train_data)
normed_test_data=norm(test_data)


# # MODEL BUILDING

# In[16]:


# Build the model
def build_model():
    model = keras.Sequential([
        layers.Dense(64, activation=tf.nn.relu, input_shape=[len(train_data.keys())]),
        layers.Dense(64, activation=tf.nn.relu),
        layers.Dense(1)
    ])
    optimizer=tf.keras.optimizers.RMSprop(0.001)
    model.compile(loss='mse',
                 optimizer=optimizer,
                 metrics=['mae','mse','accuracy'])
    return model


# In[17]:


model=build_model()


# In[19]:


# Inspect the Model
model.summary()


# In[20]:


example_batch = normed_train_data[:10]
example_result= model.predict(example_batch)
example_result


# ##### it seems to be working,and it produces a result of the expected shape and type

# In[25]:


# Train the model for 100 epoch
 
class PrintDot(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs):
        if epoch % 100 == 0: 
            print('')
            print('.', end='')
EPOCHS=100
history = model.fit(
 normed_train_data,train_labels,
 epochs=EPOCHS, validation_split = 0.2, verbose = 0,
 callbacks=[PrintDot()])


# In[26]:


hist=pd.DataFrame(history.history)
hist['epoch']=history.epoch
hist.tail()


# In[27]:


# Visualize the model's training progress
def plot_history(history):
    hist = pd.DataFrame(history.history)
    hist['epoch']=history.epoch
    
    plt.figure()
    plt.xlabel('Epoch')
    plt.ylabel('Mean Abs Error [MPG]')
    plt.plot(hist['epoch'], hist['mae'],
            label='Train Error')
    plt.plot(hist['epoch'], hist['val_mae'],
            label='Val Error')
    plt.legend()
    plt.ylim([0,5])
    
    plt.figure()
    plt.xlabel('Epoch')
    plt.ylabel('Mean Square Error [$MPG^2$]')
    plt.plot(hist['epoch'],hist['mse'],
            label='Train Error')
    plt.plot(hist['epoch'],hist['val_mse'],
            label='Val Error')
    plt.legend()
    plt.ylim([0,20])

plot_history(history)


# In[28]:


model = build_model()
# The patience parameter is the amount of epochs to check for improvement
early_stop= keras.callbacks.EarlyStopping(monitor='val_loss',patience=10)

history=model.fit (normed_train_data,train_labels,epochs=EPOCHS,
                  validation_split = 0.2, verbose=0, callbacks=[early_stop,PrintDot()])
plot_history(history)


# In[29]:


# let's see how well the model generalizes by using test set
d=model.evaluate(normed_test_data,test_labels,verbose=0)
print('Testing set Mean Abs Error: ',d[1]*100)


# #  Predict

# In[30]:


# fianlly, predict TEY values using data in the testing set:
test_prediction=model.predict(normed_test_data).flatten()
plt.scatter(test_labels, test_prediction)
plt.xlabel('True Value [TEY]')
plt.ylabel('Prediction [TEY]')
plt.axis('equal')
plt.axis('square')
plt.xlim([0,plt.xlim()[1]])
plt.ylim([0,plt.ylim()[1]])
_=plt.plot([-100,100],[-100,100])


# ##### Its looks like our model predicts good 

# In[ ]:




