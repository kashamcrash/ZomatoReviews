#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Build a model to predict the rating in a review based on the content of the text review, identify any mismatch
# Credentials: kasham1991@gmail.com / karan sharma

# Agenda
# 1. Perform specific data cleanup, 
# 2. Build a rating prediction model using the Random Forest technique and NLTK


# In[2]:


# Importing the required libraries
import pandas as pd 
import numpy as np
import re


# In[3]:


# loading the data
zomato = pd.read_csv('C://Datasets//ZomatoReviews.csv')
zomato.head()


# In[4]:


# Statistics of the dataset
# Multiple records in review_text are missing
zomato.describe(include = "all")


# 14 rows are missing the review text - need to get rid of these records

# In[5]:


# Dropping the missing values
zomato1 = zomato[~zomato.review_text.isnull()].copy()
zomato1.reset_index(inplace=True, drop=True)


# In[6]:


# Checking the shape
zomato.shape, zomato1.shape


# In[7]:


# Converting to list
zomato_list = zomato1.review_text.values
len(zomato_list)


# In[8]:


# Cleaning the text: Step-by-Step

# 1. Normalizing
# 2. Removing extra line breaks from the text
# 3. Removing stop words
# 4. Removing Punctuation


# In[9]:


# Normalizing to lower case
lower = [txt.lower() for txt in zomato_list]
lower[2:4]


# In[10]:


# Removing extra line breaks
line_break = [" ".join(txt.split()) for txt in lower]
line_break[2:4]


# In[11]:


# Tokenization
from nltk.tokenize import word_tokenize
print(word_tokenize(line_break[0]))


# In[12]:


zomato_tokens = [word_tokenize(sent) for sent in line_break]
print(line_break[0])


# In[13]:


# Removing stop words and punctuation
from nltk.corpus import stopwords
from string import punctuation


# In[14]:


stop_nltk = stopwords.words("english")
stop_punct = list(punctuation)
print(stop_nltk)


# In[15]:


# Removing no, not, don, won from stop words
# These words are important for rating 
stop_nltk.remove("no")
stop_nltk.remove("not")
stop_nltk.remove("don")
stop_nltk.remove("won")


# In[16]:


# Checking for the same
"no" in stop_nltk


# In[17]:


stop_final = stop_nltk + stop_punct + ["...", "``","''", "====", "must"]


# In[18]:


# Creating a function for the steps mentioned above
def delete(sent):
    return [term for term in sent if term not in stop_final]

delete(zomato_tokens[1])


# In[19]:


# Final clean list
zomato_clean = [delete(sent) for sent in zomato_tokens]


# In[20]:


final_clean = [" ".join(sent) for sent in zomato_clean]
final_clean[:2]
# len(final_clean)


# In[21]:


# Splitting the dataset by 70/30
from sklearn.model_selection import train_test_split
x = final_clean
y = zomato1.rating

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.30, random_state = 1)


# In[22]:


# Creating TfIdf
from sklearn.feature_extraction.text import TfidfVectorizer
vector = TfidfVectorizer(max_features = 5000)
# len(x_train), len(x_test)

x_train_bow = vector.fit_transform(x_train)
x_test_bow = vector.transform(x_test)
x_train_bow.shape, x_test_bow.shape


# In[23]:


# Model building with RF and GBR
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor


# In[24]:


model1 = RandomForestRegressor(random_state = 42, n_estimators = 10)


# In[25]:


get_ipython().run_cell_magic('time', '', 'model1.fit(x_train_bow, y_train)')


# In[26]:


# Making the prediction
# RSME score
y_train_preds = model1.predict(x_train_bow)

from sklearn.metrics import mean_squared_error
mean_squared_error(y_train, y_train_preds)**0.5


# In[27]:


# Increasing the number of trees
model2 = RandomForestRegressor(random_state = 42, n_estimators = 20)


# In[28]:


get_ipython().run_cell_magic('time', '', 'model2.fit(x_train_bow, y_train)')


# In[29]:


# RSME post 20 tress as estimators
y_train_preds = model2.predict(x_train_bow)
mean_squared_error(y_train, y_train_preds)**0.5


# In[30]:


# Finding the best hyper-parameters for the SVM classifier
# Hyperparameter tuning and GridSearch
# max_features – ‘auto’, ‘sqrt’, ‘log2’
# max_depth – 10, 15, 20, 25
from sklearn.model_selection import GridSearchCV


# In[31]:


model3 = RandomForestRegressor(random_state = 42, n_estimators = 30)


# In[32]:


# Create the parameter grid based on the results of random search 
param_grid = {
    'max_features': [500, "sqrt", "log2", "auto"],
    'max_depth': [10, 15, 20, 25]
}


# In[33]:


# Instantiate the grid search model with stratified 5 fold cross-validation scheme
grid_search = GridSearchCV(estimator = model3, param_grid = param_grid, 
                          cv = 5, n_jobs = -1, verbose = 1, scoring = "neg_mean_squared_error" )


# In[34]:


grid_search.fit(x_train_bow, y_train)


# In[35]:


grid_search.cv_results_.keys()


# In[36]:


grid_search.best_estimator_


# In[37]:


# Making predictions on the test set with RSME
# The score are higher than before
y_train_pred = grid_search.best_estimator_.predict(x_train_bow)
y_test_pred = grid_search.best_estimator_.predict(x_test_bow)


# In[38]:


mean_squared_error(y_train, y_train_pred)**0.5


# In[39]:


mean_squared_error(y_test, y_test_pred)**0.5


# In[40]:


# Looking for any mismatches
# Creating a crosstab for the same
# Calculating the difference
difference = pd.DataFrame({'review':x_test, 'rating':y_test, 'rating_pred':y_test_pred})
difference


# In[41]:


a = difference[(difference.rating - difference.rating_pred)>=2]
# a.shape
a


# In[42]:


# Thank You :) 

