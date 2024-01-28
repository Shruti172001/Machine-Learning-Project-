#!/usr/bin/env python
# coding: utf-8

# # SMS is spam or ham

# ## Importing the Dependencies

# In[18]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


# ## Data Collection & Pre-Processing

# In[19]:


# loading the data from csv file to a pandas Dataframe
raw_mail_data = pd.read_csv("C:/Users/ADMIN/Downloads/mail_data - mail_data.csv")


# In[20]:


raw_mail_data.head()


# In[21]:


# replace the null values with a null string
mail_data = raw_mail_data.where((pd.notnull(raw_mail_data)),'')


# In[22]:


# checking the number of rows and columns in the dataframe
mail_data.shape


# In[23]:


# checking the total no. of element in the dataframe
mail_data.size


# In[24]:


# checking the null values in the dataframe
mail_data.isnull().sum()


# In[25]:


mail_data.info()


# In[26]:


mail_data.dtypes


# In[27]:


mail_data["Category"].value_counts()


# In[28]:


plt.pie(mail_data["Category"].value_counts(normalize=True)*100,labels=("spam","ham"))


# # Label Encoding

# In[29]:


# label spam mail as 0;  ham mail as 1;

mail_data.loc[mail_data['Category'] == 'spam', 'Category',] = 0
mail_data.loc[mail_data['Category'] == 'ham', 'Category',] = 1


# In[ ]:





# In[30]:


# separating the data as texts and label

X = mail_data['Message']

Y = mail_data['Category']


# In[31]:


print(X)


# In[32]:


print(Y)


# ## Splitting the data into training data & test data

# In[33]:


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=3)


# In[34]:


print(X.shape)
print(X_train.shape)
print(X_test.shape)


# ## Feature Extraction

# In[35]:


# transform the text data to feature vectors that can be used as input to the Logistic regression

feature_extraction = TfidfVectorizer(min_df = 1, stop_words='english')

X_train_features = feature_extraction.fit_transform(X_train)
X_test_features = feature_extraction.transform(X_test)

# convert Y_train and Y_test values as integers

Y_train = Y_train.astype('int')
Y_test = Y_test.astype('int')


# In[36]:


print(X_train)


# In[37]:


print(X_train_features)


# # Training the Mode

# ## Logistic Regression

# In[38]:


model = LogisticRegression()


# In[39]:


# training the Logistic Regression model with the training data
model.fit(X_train_features, Y_train)


# ## Evaluating the trained model

# In[40]:


# prediction on training data

prediction_on_training_data = model.predict(X_train_features)
accuracy_on_training_data = accuracy_score(Y_train, prediction_on_training_data)


# In[41]:


print('Accuracy on training data : ', accuracy_on_training_data)


# In[42]:


# prediction on test data

prediction_on_test_data = model.predict(X_test_features)
accuracy_on_test_data = accuracy_score(Y_test, prediction_on_test_data)


# In[43]:


print('Accuracy on test data : ', accuracy_on_test_data)


# # Building a Predictive System

# In[44]:


input_mail = ["Free entry in 2 a wkly comp to win FA Cup final tkts 21st May 2005. Text FA to 87121 to receive entry question(std txt rate)T&C's apply 08452810075over18's"]

# convert text to feature vectors
input_data_features = feature_extraction.transform(input_mail)

# making prediction

prediction = model.predict(input_data_features)
print(prediction)


if (prediction[0]==1):
  print('Ham mail')

else:
  print('Spam mail')


# # The End

# In[ ]:




