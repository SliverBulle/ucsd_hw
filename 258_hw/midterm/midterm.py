#!/usr/bin/env python
# coding: utf-8

# In[1]:


import json
import gzip
import math
import numpy
from collections import defaultdict
from sklearn import linear_model
import random
import statistics


# In[2]:


def assertFloat(x):
    assert type(float(x)) == float

def assertFloatList(items, N):
    assert len(items) == N
    assert [type(float(x)) for x in items] == [float]*N


# In[3]:


answers = {}


# In[4]:


# From https://cseweb.ucsd.edu/classes/fa24/cse258-b/files/steam.json.gz
z = gzip.open("data/steam.json.gz")


# In[5]:


dataset = []
for l in z:
    d = eval(l)
    dataset.append(d)


# In[6]:


z.close()


# In[7]:


dataset[0]


# In[8]:


### Question 1


# In[9]:


from sklearn.metrics import mean_squared_error
def MSE(y, ypred):
    return mean_squared_error(y, ypred)


# In[10]:


#def feat1(d):
import numpy as np
def feat1(X,y):
    model = linear_model.LinearRegression()
    model.fit(X, y)

    return model


# In[11]:


X = np.array([[len(data["text"])] for data in dataset])
y = np.array([data['hours'] for data in dataset])  


# In[12]:


mod = feat1(X,y)


# In[13]:


mse1 = MSE(y, mod.predict(X))


# In[14]:


mod.coef_


# In[15]:


answers['Q1'] = [float(mod.coef_[0]), float(mse1)] # Remember to cast things to float rather than (e.g.) np.float64


# In[16]:


assertFloatList(answers['Q1'], 2)


# In[17]:


### Question 2


# In[18]:


dataTrain = dataset[:int(len(dataset)*0.8)]
dataTest = dataset[int(len(dataset)*0.8):]


# In[19]:


X_train = np.array([[len(data["text"])] for data in dataTrain])
y_train = np.array([data['hours'] for data in dataTrain])  
X_test = np.array([[len(data["text"])] for data in dataTest])
y_test = np.array([data['hours'] for data in dataTest])  


# In[20]:


mod2 = feat1(X_train,y_train)
y_pred = mod2.predict(X_test)
mse2 = MSE(y_test, y_pred)


# In[21]:


under = 0
over = 0

for i in range(len(y_test)):
    if y_test[i] > y_pred[i]:
        under += 1
    elif y_test[i] < y_pred[i]:
        over += 1


# In[22]:


print(under, over)


# In[23]:


answers['Q2'] = [mse2, under, over]


# In[24]:


assertFloatList(answers['Q2'], 3)


# In[25]:


### Question 3


# In[ ]:





# In[26]:


#a
y2 = y[:]
y2.sort()
perc90 = y2[int(len(y2)*0.9)] # 90th percentile

X3a = []
y3a = []

for i in range(len(X_train)):
    if y_train[i] <= perc90:
        X3a.append(X_train[i])
        y3a.append(y_train[i])
mod3a = linear_model.LinearRegression()
mod3a.fit(X3a,y3a)
pred3a = mod3a.predict(X_test)


# In[27]:


under3a = sum(y_test[i] > pred3a[i] for i in range(len(y_test)))
over3a = sum(y_test[i] <= pred3a[i] for i in range(len(y_test)))


# In[28]:


# etc. for 3b and 3c
y3b = np.array([d["hours_transformed"] for d in dataset])
X3b = X[:]

Xtrain3b = X3b[:int(len(X3b)*0.8)]
ytrain3b = y3b[:int(len(y3b)*0.8)]
Xtest3b = X3b[int(len(X3b)*0.8):]
ytest3b = y3b[int(len(y3b)*0.8):]

mod3b = linear_model.LinearRegression()
mod3b.fit(Xtrain3b,ytrain3b)
pred3b = mod3b.predict(Xtest3b)
under3b = sum(ytest3b[i] >= pred3b[i] for i in range(len(ytest3b)))
over3b = sum(ytest3b[i] < pred3b[i] for i in range(len(ytest3b)))


# In[29]:


#3c
theta0 = mod2.intercept_  


median_length = np.median([len(d["text"]) for d in dataTrain])
median_hours = np.median([d["hours"] for d in dataTrain])

theta1 = (median_hours - theta0) / median_length

pred3c = theta0 + theta1 * X_test.reshape(-1)


under3c = sum(y_test[i] >= pred3c[i] for i in range(len(y_test)))
over3c = sum(y_test[i] < pred3c[i] for i in range(len(y_test)))


# In[30]:


answers['Q3'] = [under3a, over3a, under3b, over3b, under3c, over3c]


# In[31]:


assertFloatList(answers['Q3'], 6)


# In[32]:


### Question 4


# In[33]:


y = np.array([1 if d["hours"] > median_hours else 0 for d in dataset])
ytest = np.array([1 if d["hours"] > median_hours else 0 for d in dataTest])

X = np.array([[len(d["text"])] for d in dataset])
Xtest = np.array([[len(d["text"])] for d in dataTest])


# In[34]:


mod4 = linear_model.LogisticRegression(C=1)
mod4.fit(X,y)
predictions = mod4.predict(Xtest) # Binary vector of predictions


# In[35]:


TP = sum((predictions == 1) & (ytest == 1))
TN = sum((predictions == 0) & (ytest == 0))
FP = sum((predictions == 1) & (ytest == 0))
FN = sum((predictions == 0) & (ytest == 1))
BER = (FP + FN) / (TP + TN + FP + FN)
print(TP, TN, FP, FN, BER)


# In[36]:


answers['Q4'] = [TP, TN, FP, FN, BER]


# In[37]:


assertFloatList(answers['Q4'], 5)


# In[38]:


### Question 5


# In[39]:


under5 = FP
over5 = FN


# In[40]:


answers['Q5'] = [under5, over5]


# In[41]:


assertFloatList(answers['Q5'], 2)


# In[42]:


### Question 6


# In[43]:


from sklearn.metrics import confusion_matrix
#6a
X2014 = np.array([[len(d["text"])] for d in dataset if int(d["date"][:4]) <= 2014])
y2014 = np.array([1 if d["hours"] > median_hours else 0 for d in dataset if int(d["date"][:4]) <= 2014])

X2014train = X2014[:int(len(X2014)*0.8)]
y2014train = y2014[:int(len(y2014)*0.8)]
X2014test = X2014[int(len(X2014)*0.8):]
y2014test = y2014[int(len(y2014)*0.8):]

mod6a = linear_model.LogisticRegression(C=1)
mod6a.fit(X2014train,y2014train)
pred6a = mod6a.predict(X2014test)

cm6a = confusion_matrix(y2014test, pred6a)
BER_A = (cm6a[0][1] + cm6a[1][0]) / (cm6a[0][0] + cm6a[1][1] + cm6a[0][1] + cm6a[1][0])


# In[44]:


#6b
X2015 = np.array([[len(d["text"])] for d in dataset if int(d["date"][:4]) > 2014])
y2015 = np.array([1 if d["hours"] > median_hours else 0 for d in dataset if int(d["date"][:4]) > 2014])

X2015train = X2015[:int(len(X2015)*0.8)]
y2015train = y2015[:int(len(y2015)*0.8)]
X2015test = X2015[int(len(X2015)*0.8):]
y2015test = y2015[int(len(y2015)*0.8):]

mod6b = linear_model.LogisticRegression(C=1)
mod6b.fit(X2015train,y2015train)
pred6b = mod6b.predict(X2015test)
cm6b = confusion_matrix(y2015test, pred6b)
BER_B = (cm6b[0][1] + cm6b[1][0]) / (cm6b[0][0] + cm6b[1][1] + cm6b[0][1] + cm6b[1][0])

#6c
mod6c = linear_model.LogisticRegression(C=1)
mod6c.fit(X2014,y2014)
pred6c = mod6c.predict(X2015)
cm6c = confusion_matrix(y2015, pred6c)
BER_C = (cm6c[0][1] + cm6c[1][0]) / (cm6c[0][0] + cm6c[1][1] + cm6c[0][1] + cm6c[1][0])

#6d
mod6d = linear_model.LogisticRegression(C=1)
mod6d.fit(X2015,y2015)
pred6d = mod6d.predict(X2014)
cm6d = confusion_matrix(y2014, pred6d)
BER_D = (cm6d[0][1] + cm6d[1][0]) / (cm6d[0][0] + cm6d[1][1] + cm6d[0][1] + cm6d[1][0])


# In[45]:


answers['Q6'] = [BER_A, BER_B, BER_C, BER_D]


# In[46]:


assertFloatList(answers['Q6'], 4)


# In[47]:


### Question 7


# In[48]:


usersPerItem = defaultdict(set) # Maps an item to the users who rated it
itemsPerUser = defaultdict(set) # Maps a user to the items that they rated
reviewsPerUser = defaultdict(list)
reviewsPerItem = defaultdict(list)



for d in dataTrain:
    usersPerItem[d["gameID"]].add(d["userID"])
    itemsPerUser[d["userID"]].add(d["gameID"])
    reviewsPerUser[d["userID"]].append(d)
    reviewsPerItem[d["gameID"]].append(d)


# In[49]:


def Jaccard(s1, s2):
    numer = len(s1.intersection(s2))
    denom = len(s1.union(s2))
    if denom == 0:
        return 0
    return numer / denom


# In[50]:


def similarity(user, N):
    similarities = []
    items = itemsPerUser[user]
    for u2 in itemsPerUser:
        if u2 == user: continue
        sim = Jaccard(items, itemsPerUser[u2])
        similarities.append((sim,u2))
    similarities.sort(reverse=True)
    return similarities[:N]


# In[51]:


sim = similarity(dataset[0]["userID"], 10)
first = sim[0][0]
tenth = sim[9][0]

answers['Q7'] = [first, tenth]


# In[52]:


assertFloatList(answers['Q7'], 2)


# In[53]:


### Question 8


# In[54]:


global_mean = np.mean([d["hours_transformed"] for d in dataTrain])


# In[55]:


def predict_user_based(user, item):
    if user not in itemsPerUser or item not in usersPerItem:
        return global_mean
    hours  = []
    similarities  = []
    #v is users who comment on game i 
    for d in reviewsPerItem[item]:
        v = d["userID"]
        if v == user:
            continue
        hours.append(d["hours_transformed"])
        similarities.append(Jaccard(itemsPerUser[user],itemsPerUser[v]))

    if (sum(similarities) > 0):
        weightedRatings = [(x*y) for x,y in zip(hours,similarities)]
        return sum(weightedRatings) / sum(similarities)
    else:
        return global_mean



# In[56]:


def predict_item_based(item, user):
    if user not in itemsPerUser or item not in usersPerItem:
        return global_mean
    hours = []
    similarities = []
    for d in reviewsPerUser[user]:
        i2 = d["gameID"]
        hours.append(d["hours_transformed"])
        similarities.append(Jaccard(usersPerItem[item],usersPerItem[i2]))
    if (sum(similarities) > 0):
        weightedRatings = [(x*y) for x,y in zip(hours,similarities)]
        return sum(weightedRatings) / sum(similarities)
    else:
        return global_mean
predictionsUser = []
actualsUser = []
for d in dataTest:
    pred = predict_user_based(d["userID"], d["gameID"])
    predictionsUser.append(pred)
    actualsUser.append(d["hours_transformed"])

MSEU = mean_squared_error(actualsUser, predictionsUser)

predictionsItem = []
actualsItem = []
for d in dataTest:
    pred = predict_item_based(d["gameID"], d["userID"])
    predictionsItem.append(pred)
    actualsItem.append(d["hours_transformed"])
MSEI = mean_squared_error(actualsItem, predictionsItem)


# In[57]:


answers['Q8'] = [MSEU, MSEI]


# In[58]:


assertFloatList(answers['Q8'], 2)


# In[59]:


### Question 9


# In[73]:


def predict_time_weighted(user, item):
    if user not in itemsPerUser or item not in usersPerItem:
        return global_mean
    hours  = []
    similarities  = []
    e_list = []
    for d in reviewsPerItem[item]:
        v = d["userID"]
        if v == user:
            continue
        hours.append(d["hours_transformed"])
        similarities.append(Jaccard(itemsPerUser[user],itemsPerUser[v]))
        data_u = [int(reviewsPerUser[user][i]["date"][:4]) for i in range(len(reviewsPerUser[user])) if reviewsPerUser[user][i]["gameID"] == item]        
        if not data_u:
            data_u = [int(d["date"][:4])]
        
        e = math.exp(-abs(data_u[0] - int(d["date"][:4])))
        e_list.append(e)

    if (sum(similarities) > 0):
        weightedRatings = [(x*y*e) for x,y,e in zip(hours,similarities,e_list)]
        similarities = [e*y for e,y in zip(e_list,similarities)]
        return sum(weightedRatings) / sum(similarities)
    else:
        return global_mean


# 

# In[74]:


predictions = []
actuals = []
for d in dataTest:
    pred = predict_time_weighted(d["userID"], d["gameID"])
    predictions.append(pred)
    actuals.append(d["hours_transformed"])

MSE9 = mean_squared_error(actuals, predictions)
answers['Q9'] = MSE9


# In[212]:


answers['Q9'] = MSE9


# In[213]:


assertFloat(answers['Q9'])


# In[228]:


if "float" in str(answers) or "int" in str(answers):
    print("it seems that some of your answers are not native python ints/floats;")
    print("the autograder will not be able to read your solution unless you convert them to ints/floats")


# In[229]:


f = open("answers_midterm.txt", 'w')
f.write(str(answers) + '\n')
f.close()


# In[ ]:




