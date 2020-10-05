#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd               # dataset upload
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

get_ipython().run_line_magic('matplotlib', 'inline')
house = pd.read_csv('USA_Housing.csv')

house.head() 


# In[2]:


house.info()      # information about the dataset


# In[3]:


house.describe()          #statistical description of the dataset


# In[4]:


house.columns       # columns


# In[5]:


sns.pairplot(house)        #visualizing with respect to different axes


# In[6]:


sns.distplot(house['Price'])
sns.heatmap(house.corr(), annot=True)   #visualizing data


# In[7]:


X = house[['Avg. Area Income', 'Avg. Area House Age', 'Avg. Area Number of Rooms',
               'Avg. Area Number of Bedrooms', 'Area Population']]

y = house['Price']                # setting x & y for splitting


# In[8]:


from sklearn.model_selection import train_test_split                   #applying train_test sklearn

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=101) 


# In[9]:


from sklearn.linear_model import LinearRegression                #performing linear regression

lm = LinearRegression() 

lm.fit(X_train,y_train) 


# In[10]:


print(lm.intercept_)           #intercept


# In[12]:


predictions = lm.predict(X_test)             #prediction using scatter plot
plt.scatter(y_test,predictions)


# In[13]:


sns.distplot((y_test-predictions),bins=50);   #how well our model performed is showed by histogram


# In[ ]:




