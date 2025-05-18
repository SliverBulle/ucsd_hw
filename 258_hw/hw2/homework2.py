#!/usr/bin/env python
# coding: utf-8

# In[113]:


import numpy
import urllib
import scipy.optimize
import random
from sklearn import linear_model
import gzip
from collections import defaultdict


# In[114]:


import warnings
warnings.filterwarnings("ignore")


# In[115]:


def assertFloat(x):
    assert type(float(x)) == float

def assertFloatList(items, N):
    assert len(items) == N
    assert [type(float(x)) for x in items] == [float]*N


# In[116]:


import zipfile
import os

def unzip_file(zip_file_path, extract_to_folder):
    if not os.path.exists(extract_to_folder):
        os.makedirs(extract_to_folder)
    with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to_folder)

unzip_file('polish+companies+bankruptcy+data.zip', 'data')


# In[117]:


f = open("data/5year.arff", 'r')


# In[118]:


# Read and parse the data
while not '@data' in f.readline():
    pass

dataset = []
for l in f:
    if '?' in l: # Missing entry
        continue
    l = l.split(',')
    values = [1] + [float(x) for x in l]
    values[-1] = values[-1] > 0 # Convert to bool
    dataset.append(values)


# In[119]:


X = [d[:-1] for d in dataset]
y = [d[-1] for d in dataset]


# In[120]:


answers = {} # Your answers


# In[121]:


def accuracy(predictions, y):
    correct = sum(p == actual for p, actual in zip(predictions, y))
    return correct / len(y)    


# In[122]:


from sklearn.metrics import confusion_matrix
def BER(predictions, y):
    tn, fp, fn, tp = confusion_matrix(y, predictions).ravel()

    fpr = fp / (fp + tn)  # False Positive Rate
    fnr = fn / (fn + tp)  # False Negative Rate
    ber = (fpr + fnr) / 2
    return ber


# In[123]:


### Question 1


# In[124]:


mod = linear_model.LogisticRegression(C=1)
mod.fit(X,y)

pred = mod.predict(X)

acc1 = accuracy(pred, y)
ber1 = BER(pred, y)


# In[125]:


answers['Q1'] = [acc1, ber1] # Accuracy and balanced error rate


# In[126]:


assertFloatList(answers['Q1'], 2)


# In[127]:


### Question 2


# In[128]:


mod = linear_model.LogisticRegression(C=1, class_weight='balanced')
mod.fit(X,y)

pred = mod.predict(X)


# In[129]:


acc2 = accuracy(pred, y)
ber2 = BER(pred, y)

answers['Q2'] = [acc2, ber2]


# In[130]:


answers['Q2'] = [acc2, ber2]


# In[131]:


assertFloatList(answers['Q2'], 2)


# In[132]:


### Question 3


# In[133]:


random.seed(3)
random.shuffle(dataset)


# In[134]:


X = [d[:-1] for d in dataset]
y = [d[-1] for d in dataset]


# In[135]:


Xtrain, Xvalid, Xtest = X[:len(X)//2], X[len(X)//2:(3*len(X))//4], X[(3*len(X))//4:]
ytrain, yvalid, ytest = y[:len(X)//2], y[len(X)//2:(3*len(X))//4], y[(3*len(X))//4:]


# In[136]:


len(Xtrain), len(Xvalid), len(Xtest)


# In[137]:


mod = linear_model.LogisticRegression(C=1.0, class_weight='balanced')
mod.fit(Xtrain, ytrain)

# predict
pred_train = mod.predict(Xtrain)
pred_valid = mod.predict(Xvalid)
pred_test = mod.predict(Xtest)

# calculate BER
berTrain = BER(pred_train, ytrain)
berValid = BER(pred_valid, yvalid)
berTest = BER(pred_test, ytest)


# In[138]:


answers['Q3'] = [berTrain, berValid, berTest]


# In[139]:


assertFloatList(answers['Q3'], 3)


# In[140]:


### Question 4


# In[141]:


C_values = [10**i for i in range(-4, 5)]
berList = []

for C in C_values:
    mod = linear_model.LogisticRegression(C=C, class_weight='balanced')
    mod.fit(Xtrain, ytrain)
    pred_valid = mod.predict(Xvalid)
    ber = BER(pred_valid, yvalid)
    berList.append(ber)


# In[142]:


answers['Q4'] = berList


# In[143]:


assertFloatList(answers['Q4'], 9)


# In[144]:


### Question 5


# In[145]:


best_index = berList.index(min(berList))
bestC = C_values[best_index]

mod = linear_model.LogisticRegression(C=bestC, class_weight='balanced')
mod.fit(Xtrain, ytrain)
pred_test = mod.predict(Xtest)
ber5 = BER(pred_test, ytest)


# In[146]:


answers['Q5'] = [bestC, ber5]


# In[147]:


assertFloatList(answers['Q5'], 2)


# In[148]:


### Question 6


# In[149]:


f = gzip.open("young_adult_10000.json.gz")
dataset = []
for l in f:
    dataset.append(eval(l))


# In[150]:


dataTrain = dataset[:9000]
dataTest = dataset[9000:]


# In[151]:


dataTrain[0]


# In[152]:


# Some data structures you might want

usersPerItem = defaultdict(set) # Maps an item to the users who rated it
itemsPerUser = defaultdict(set) # Maps a user to the items that they rated
reviewsPerUser = defaultdict(list)
reviewsPerItem = defaultdict(list)
ratingDict = {} # To retrieve a rating for a specific user/item pair

for d in dataTrain:
    usersPerItem[d['book_id']].add(d['user_id'])
    itemsPerUser[d['user_id']].add(d['book_id'])
    reviewsPerUser[d['user_id']].append(d)
    reviewsPerItem[d['book_id']].append(d)
    ratingDict[(d['user_id'], d['book_id'])] = d['rating']


# In[153]:


def Jaccard(s1, s2):
    numer = len(s1.intersection(s2))
    denom = len(s1.union(s2))
    if denom == 0:
        return 0
    return numer / denom


# In[154]:


def mostSimilar(i, N):
    similarities = []
    users = usersPerItem[i]
    for i2 in usersPerItem:
        if i2 == i: continue
        sim = Jaccard(users, usersPerItem[i2])
        #sim = Pearson(i, i2) # Could use alternate similarity metrics straightforwardly
        similarities.append((sim,i2))
    similarities.sort(reverse=True)
    return similarities[:N]


# In[ ]:





# In[155]:


answers['Q6'] = mostSimilar('2767052', 10)


# In[156]:


assert len(answers['Q6']) == 10
assertFloatList([x[0] for x in answers['Q6']], 10)


# In[157]:


### Question 7


# In[158]:


userAverages = {}
itemAverages = {}

for u in itemsPerUser:
    rs = [ratingDict[(u,i)] for i in itemsPerUser[u]]
    userAverages[u] = sum(rs) / len(rs)
    
for i in usersPerItem:
    rs = [ratingDict[(u,i)] for u in usersPerItem[i]]
    itemAverages[i] = sum(rs) / len(rs)


# In[159]:


ratingMean = sum([d['rating'] for d in dataset]) / len(dataset)


# In[160]:


def predict_rating(user, item) -> float:
    #rated_items = itemsPerUser[user]
    ratings  = []
    similarities  = []
    for d in reviewsPerUser[user]:
        i2 = d['book_id']
        if i2 == item:
            continue
        ratings.append(d['rating'] - itemAverages[i2])
        similarities.append(Jaccard(usersPerItem[item],usersPerItem[i2]))

    if (sum(similarities) > 0):
        weightedRatings = [(x*y) for x,y in zip(ratings,similarities)]
        return itemAverages[item] + sum(weightedRatings) / sum(similarities)
    else:
        # User hasn't rated any similar items
        return ratingMean

from sklearn.metrics import mean_squared_error

predictions = []
actuals = []
for d in dataTest:
    user = d['user_id']
    item = d['book_id']
    actual = d['rating']
    pred = predict_rating(user, item)
    predictions.append(pred)
    actuals.append(actual)

mse7 = mean_squared_error(actuals, predictions)


# In[161]:


answers['Q7'] = mse7


# In[162]:


assertFloat(answers['Q7'])


# In[163]:


### Question 8


# In[164]:


def predict_rating_user_based(user, item) -> float:
    #rated_items = itemsPerUser[user]
    ratings  = []
    similarities  = []
    for d in reviewsPerItem[item]:
        v = d['user_id']
        if v == user:
            continue
        ratings.append(d['rating'] - userAverages[v])
        similarities.append(Jaccard(itemsPerUser[user],itemsPerUser[v]))

    if (sum(similarities) > 0):
        weightedRatings = [(x*y) for x,y in zip(ratings,similarities)]
        return itemAverages[item] + sum(weightedRatings) / sum(similarities)
    else:
        # User hasn't rated any similar items
        return ratingMean

from sklearn.metrics import mean_squared_error

predictions = []
actuals = []
for d in dataTest:
    user = d['user_id']
    item = d['book_id']
    actual = d['rating']
    pred = predict_rating_user_based(user, item)
    predictions.append(pred)
    actuals.append(actual)

mse8 = mean_squared_error(actuals, predictions)


# In[110]:


answers['Q8'] = mse8


# In[111]:


assertFloat(answers['Q8'])


# In[112]:


f = open("answers_hw2.txt", 'w')
f.write(str(answers) + '\n')
f.close()


# In[ ]:




