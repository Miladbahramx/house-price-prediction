#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error , r2_score


# In[2]:


data = pd.read_csv(r'D:/Python/DATA/New/House_Price_Prediction_Dataset.csv')


# In[3]:


data.head()


# In[4]:


data.info()


# In[9]:


data['Garage'] = data['Garage'].map({ 'Yes': 1, 'No': 0 })


# In[10]:


X = data[['Area' , 'Bedrooms' , 'YearBuilt' , 'Garage']]
Y = data['Price']


# In[11]:


X_train,X_test,Y_train,Y_test = train_test_split(X , Y , test_size=0.2 , random_state=42)


# In[12]:


model = LinearRegression()
model.fit(X_train , Y_train)


# In[13]:


predictions = model.predict(X_test)


# In[14]:


mae = mean_absolute_error(Y_test , predictions)
r2 = r2_score(Y_test , predictions)


# In[50]:


new_House = pd.DataFrame({
    'Area':[1000] ,
    'Bedrooms':[1] ,
    'YearBuilt':[2000] ,
    'Garage':[1]
})


# In[51]:


predicted_price = model.predict(new_House)


# In[52]:


print('Predicred Price :' , int(predicted_price) ,'$')


# In[53]:


plt.scatter(Y_test, predictions)

plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Actual vs Predicted House Prices")

plt.show()


# In[ ]:




