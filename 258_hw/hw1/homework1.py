#!/usr/bin/env python
# coding: utf-8

# In[1]:


import json
from matplotlib import pyplot as plt
from collections import defaultdict
from sklearn import linear_model
import numpy as np
import random
import gzip
import math


# In[2]:


import warnings
warnings.filterwarnings("ignore")


# In[3]:


def assertFloat(x): # Checks that an answer is a float
    assert type(float(x)) == float

def assertFloatList(items, N):
    assert len(items) == N
    assert [type(float(x)) for x in items] == [float]*N


# In[4]:


f = gzip.open("young_adult_10000.json.gz")
dataset = []
for l in f:
    dataset.append(json.loads(l))


# In[5]:


len(dataset)


# In[6]:


answers = {} 


# In[7]:


dataset[0]


# In[8]:


type(dataset[0])


# ### Question 1

# In[9]:


from typing import List


def feature(datum):
    X = np.array([[review['review_text'].count('!')] for review in datum])
    y = np.array([review['rating'] for review in datum])

    model = linear_model.LinearRegression()
    model.fit(X, y)

    return model

model = feature(dataset)
theta1 = model.coef_
theta0 = model.intercept_
print(theta0, theta1)
X = np.array([[review['review_text'].count('!')] for review in dataset])
y = np.array([review['rating'] for review in dataset])

y_pred = model.predict(X)

from sklearn.metrics import mean_squared_error
mse = mean_squared_error(y, y_pred)

answers['Q1'] = [theta0, theta1[0], mse]


# In[10]:


assertFloatList(answers['Q1'], 3) # Check the format of your answer (three floats)


# ### Question 2

# In[11]:


def feature(datum):

    X = np.array([[len(review['review_text']), review['review_text'].count('!')] for review in datum]) 
    y = np.array([review['rating'] for review in datum])

    model = linear_model.LinearRegression()
    model.fit(X, y)

    return model
model = feature(dataset)
theta1 = model.coef_
theta0 = model.intercept_
print(theta0, theta1)

X = np.array([[len(review['review_text']), review['review_text'].count('!')] for review in dataset]) 
y = np.array([review['rating'] for review in dataset])

y_pred = model.predict(X)

from sklearn.metrics import mean_squared_error

mse = mean_squared_error(y, y_pred)

answers['Q2'] = [theta0, theta1[0],theta1[1], mse]


# In[12]:


assertFloatList(answers['Q2'], 4)


# ### Question 3

# In[13]:


from sklearn.preprocessing import PolynomialFeatures
def feature(datum, deg):

    #mses 
    mses = []
    # feature for a specific polynomial degree
    X = np.array([[review['review_text'].count('!')] for review in datum])
    y = np.array([review['rating'] for review in datum])
    for i in range(1, deg + 1):
        poly = PolynomialFeatures(degree = i)  
        X_poly = poly.fit_transform(X)        

        #train model
        model = linear_model.LinearRegression()
        model.fit(X_poly, y)
        #predict
        y_pred = model.predict(X_poly)

        # mse
        mse = mean_squared_error(y, y_pred)
        mses.append(mse)
    return mses

mses = feature(dataset, 5)

answers['Q3'] = mses



# In[14]:


assertFloatList(answers['Q3'], 5)# List of length 5


# ### Question 4

# In[15]:


X = np.array([[review['review_text'].count('!')] for review in dataset])
y = np.array([review['rating'] for review in dataset])

#split data into training and testing
X_train = X[:5000]
X_test = X[5000:]
y_train = y[:5000]
y_test = y[5000:]

def feature(X,y,X_test,y_test, deg):
    mses = []
    for i in range(1, deg + 1):
        poly = PolynomialFeatures(degree = i)  
        X_poly = poly.fit_transform(X_train)        

        #train model
        model = linear_model.LinearRegression()
        model.fit(X_poly, y_train)
        #predict
        X_poly_test = poly.fit_transform(X_test)
        y_pred = model.predict(X_poly_test)

        # mse
        mse = mean_squared_error(y_test, y_pred)
        mses.append(mse)
    return mses

mses = feature(X_train, y_train, X_test, y_test, 5)


# In[16]:


answers['Q4'] = mses


# In[17]:


assertFloatList(answers['Q4'], 5)


# ### Question 5

# In[18]:


from sklearn.metrics import mean_absolute_error
y_pred = model.predict(X_test)

# 计算MAE
mae = np.mean(np.abs(y_test - y_pred))

answers['Q5'] = mae
assertFloat(answers['Q5'])


# ### Question 6

# In[19]:


f = open("beer_50000.json")
dataset = []
for l in f:
    if 'user/gender' in l:
        dataset.append(eval(l))


# In[20]:


len(dataset)


# In[21]:


for k,v in dataset[0].items():
    print(k,v)


# In[22]:


X = np.array([[review['review/text'].count('!')] for review in dataset])
y = np.array([1 if review['user/gender'] == 'Female' else 0 for review in dataset])



def feature(X,y):
    model = linear_model.LogisticRegression()
    model.fit(X, y)
    return model

model = feature(X,y)
y_pred = model.predict(X)

from sklearn.metrics import confusion_matrix
tn, fp, fn, tp = confusion_matrix(y, y_pred).ravel()

fpr = fp / (fp + tn)  # False Positive Rate
fnr = fn / (fn + tp)  # False Negative Rate
ber = (fpr + fnr) / 2

answers['Q6'] = [tp,tn,fp,fn,ber]
assertFloatList(answers['Q6'], 5)


# ### Question 7

# In[23]:


#Retrain the regressor using the class weight=’balanced’ option, 
# and report the same error metrics as above.

def feature(X,y):
    model = linear_model.LogisticRegression(class_weight='balanced')
    model.fit(X, y)
    return model

model = feature(X,y)
y_pred = model.predict(X)
#print(y_pred[0:10])
from sklearn.metrics import confusion_matrix
tn, fp, fn, tp = confusion_matrix(y, y_pred).ravel()

fpr = fp / (fp + tn)  # False Positive Rate
fnr = fn / (fn + tp)  # False Negative Rate
ber = (fpr + fnr) / 2

answers['Q7'] = [tp,tn,fp,fn,ber]
assertFloatList(answers['Q7'], 5)


# ### Question 8

# In[24]:


#Report the precision@K of your balanced classifier for K ∈ [1, 10, 100, 1000, 10000] (your answer should
#be a list of five precision values).
y_probs = model.predict_proba(X)[:, 1]  # p(y=1|x)

y_pred = model.predict(X)

# precision @ K
precision_at_k = []
ks = [1, 10, 100, 1000, 10000]

for k in ks:
    # top k predictions
    indices = np.argsort(y_probs)[-k:]  
    true_positives = np.sum(y[indices] == 1)  
    precision = true_positives / k  # precision@k
    precision_at_k.append(precision)

# precision@k
print(precision_at_k)

answers['Q8'] = precision_at_k
assertFloatList(answers['Q8'], 5)


# In[25]:


f = open("answers_hw1.txt", 'w') # Write your answers to a file
f.write(str(answers) + '\n')
f.close()


# In[26]:


for k,v in answers.items():
    print(k,v)


# In[ ]:




