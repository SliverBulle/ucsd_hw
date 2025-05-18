#!/usr/bin/env python
# coding: utf-8

# In[1]:


import gzip
from collections import defaultdict
import math
import scipy.optimize
from sklearn import svm
import numpy
import string
import random
import string
from sklearn import linear_model


# In[2]:


import warnings
warnings.filterwarnings("ignore")


# In[3]:


def assertFloat(x):
    assert type(float(x)) == float

def assertFloatList(items, N):
    assert len(items) == N
    assert [type(float(x)) for x in items] == [float]*N


# In[4]:


def readGz(path):
    for l in gzip.open(path, 'rt'):
        yield eval(l)


# In[5]:


def readCSV(path):
    f = gzip.open(path, 'rt')
    f.readline()
    for l in f:
        u,b,r = l.strip().split(',')
        r = int(r)
        yield u,b,r


# In[6]:


answers = {}


# In[7]:


# Some data structures that will be useful


# In[8]:


allRatings = []
for l in readCSV("train_Interactions.csv.gz"):
    allRatings.append(l)


# In[9]:


len(allRatings)


# In[10]:


ratingsTrain = allRatings[:190000]
ratingsValid = allRatings[190000:]
ratingsPerUser = defaultdict(list)
ratingsPerItem = defaultdict(list)
usersPerItem = defaultdict(set)
itemsPerUser = defaultdict(set)
for u,b,r in ratingsTrain:
    ratingsPerUser[u].append((b,r))
    ratingsPerItem[b].append((u,r))
    usersPerItem[b].add(u)
    itemsPerUser[u].add(b)


# In[11]:


##################################################
# Read prediction                                #
##################################################


# In[12]:


# Copied from baseline code
bookCount = defaultdict(int)
totalRead = 0

for user,book,_ in readCSV("train_Interactions.csv.gz"):
    bookCount[book] += 1
    totalRead += 1

mostPopular = [(bookCount[x], x) for x in bookCount]
mostPopular.sort()
mostPopular.reverse()

return1 = set()
count = 0
for ic, i in mostPopular:
    count += ic
    return1.add(i)
    if count > totalRead/2: break


# In[13]:


### Question 1


# In[14]:


# 构建验证集
validPosData = ratingsValid # 正样本
validNegData = [] # 负样本

# 为每个正样本创建一个对应的负样本
for u,b,r in validPosData:
    # 获取该用户已读过的所有书
    readBooks = set(b for b,r in ratingsPerUser[u])
    
    # 随机选择一本用户没读过的书
    while True:
        # 从所有书中随机选择一本
        negBook = random.choice(list(bookCount.keys()))
        # 确保这本书用户没读过
        if negBook not in readBooks:
            validNegData.append((u, negBook, 0))
            break

# 合并正负样本
validWithNeg = validPosData + validNegData

# 评估基线模型性能
correct = 0
total = len(validWithNeg)

for u,b,r in validWithNeg:
    # Make prediction using return1 baseline model
    prediction = 1 if b in return1 else 0
    # Check if prediction is correct
    if prediction == (r > 0):
        correct += 1

acc1 = correct / total


# In[15]:


len(validWithNeg)


# In[16]:


answers['Q1'] = acc1


# In[17]:


assertFloat(answers['Q1'])


# In[18]:


### Question 2


# In[19]:


best_threshold = 0
best_acc = 0
best_popular_set = None
# test 10% to 90%
for threshold_ratio in numpy.arange(0.1, 0.91, 0.05):
    popular_set = set()
    count = 0
    threshold = totalRead * threshold_ratio
    
    # 构建popular_set
    for ic, i in mostPopular:
        count += ic
        popular_set.add(i)
        if count > threshold:
            break
    
    # 在验证集上评估
    correct = 0
    for u,b,r in validWithNeg:
        prediction = 1 if b in popular_set else 0
        if prediction == (r > 0):
            correct += 1
    
    acc = correct / len(validWithNeg)
    
    # 更新最佳结果
    if acc > best_acc:
        best_acc = acc
        best_threshold = threshold_ratio
        best_popular_set = popular_set

# 保存最佳阈值和准确率
threshold = best_threshold
acc2 = best_acc


# In[20]:


prediction1 = []
for u,b,r in validWithNeg:
    prediction = 1 if b in best_popular_set else 0
    prediction1.append(prediction)


# In[21]:


answers['Q2'] = [threshold, acc2]


# In[22]:


assertFloat(answers['Q2'][0])
assertFloat(answers['Q2'][1])


# In[23]:


### Question 3/4


# In[24]:


def Jaccard(s1, s2):
    return len(s1.intersection(s2)) / len(s1.union(s2))


# In[26]:


# 测试不同的阈值
thresholds = numpy.arange(0.0001, 0.2, 0.100)  # 更细致的阈值范围
best_predictions = []
best_acc = 0
from tqdm import tqdm
for threshold in tqdm(thresholds):
    correct = 0
    current_predictions = []  # 当前阈值下的预测结果
    
    for u, b, r in validWithNeg:
        # 获取用户已读的所有书
        user_books = itemsPerUser[u]
        
        # 计算最大 Jaccard 相似度
        max_sim = 0
        for read_book in user_books:
            sim = Jaccard(usersPerItem[b], usersPerItem[read_book])
            max_sim = max(max_sim, sim)
        
        # 基于阈值进行预测
        prediction = 1 if max_sim > threshold else 0
        current_predictions.append(prediction)
        if prediction == (r > 0):
            correct += 1

    acc = correct / len(validWithNeg)
    
    # 更新最佳结果
    if acc > best_acc:
        best_acc = acc
        best_threshold = threshold
        best_predictions = current_predictions.copy()  # 保存最佳预测结果

# 保存最佳准确率和预测结果
acc3 = best_acc
prediction3 = best_predictions  # 这个将用于Q4

print(f"Best threshold: {best_threshold}")
print(f"Best accuracy: {acc3}")
print(f"Number of predictions: {len(prediction3)}")


# In[357]:


#Q4
X = numpy.array([(p1,p3) for p1,p3 in zip(prediction1, prediction3)])
y = numpy.array([(r > 0) for _,_,r in validWithNeg])

import sklearn
model = sklearn.linear_model.LogisticRegression(class_weight='balanced')
model.fit(X, y)

y_pred = model.predict(X)
acc4 = numpy.mean(y_pred == y)  


# In[ ]:





# In[360]:


print(acc3, acc4)


# In[352]:


answers['Q3'] = acc3
answers['Q4'] = acc4


# In[282]:


assertFloat(answers['Q3'])
assertFloat(answers['Q4'])


# In[283]:


predictions = open("predictions_Read.csv", 'w')
for l in open("pairs_Read.csv"):
    if l.startswith("userID"):
        predictions.write(l)
        continue
    u,b = l.strip().split(',')
    # (etc.)

predictions.close()


# In[284]:


answers['Q5'] = "I confirm that I have uploaded an assignment submission to gradescope"


# In[285]:


assert type(answers['Q5']) == str


# In[286]:


##################################################
# Rating prediction                              #
##################################################


# In[287]:


users = list(set(u for u,_,_ in ratingsTrain))
items = list(set(b for _,b,_ in ratingsTrain))
len(users)


# In[288]:


from scipy.sparse import lil_matrix
from sklearn.linear_model import Ridge

def fit_bias_model_sklearn_sparse(train_data, valid_data, lambda_reg=1.0):
    # 构建用户和物品的索引映射
    users = list(set(u for u,_,_ in train_data))
    items = list(set(b for _,b,_ in train_data))
    user_to_idx = {u:i for i,u in enumerate(users)}
    item_to_idx = {b:i for i,b in enumerate(items)}
    
    n_users = len(users)
    n_items = len(items)
    
    # 使用稀疏矩阵构建训练数据
    X_train = lil_matrix((len(train_data), n_users + n_items))
    y_train = numpy.zeros(len(train_data))
    
    for i, (u,b,r) in enumerate(train_data):
        X_train[i, user_to_idx[u]] = 1  # 用户one-hot编码
        X_train[i, n_users + item_to_idx[b]] = 1  # 物品one-hot编码
        y_train[i] = r
    
    # 训练模型
    model = Ridge(alpha=lambda_reg, fit_intercept=True, solver='sag')
    model.fit(X_train, y_train)
    
    # 使用稀疏矩阵构建验证数据
    X_valid = lil_matrix((len(valid_data), n_users + n_items))
    y_valid = numpy.zeros(len(valid_data))
    
    for i, (u,b,r) in enumerate(valid_data):
        if u in user_to_idx and b in item_to_idx:
            X_valid[i, user_to_idx[u]] = 1
            X_valid[i, n_users + item_to_idx[b]] = 1
        y_valid[i] = r
    
    y_pred = model.predict(X_valid)
    valid_mse = numpy.mean((y_valid - y_pred) ** 2)
    
    return valid_mse, model.intercept_, model.coef_[:n_users], model.coef_[n_users:]

# 运行模型
validMSE, alpha, beta_user, beta_item = fit_bias_model_sklearn_sparse(ratingsTrain, ratingsValid)


# In[289]:


### Question 6


# In[ ]:





# In[290]:


answers['Q6'] = validMSE


# In[291]:


assertFloat(answers['Q6'])


# In[292]:


### Question 7


# In[293]:


max_beta_user_idx = numpy.argmax(beta_user)
min_beta_user_idx = numpy.argmin(beta_user)

maxUser = users[max_beta_user_idx]
minUser = users[min_beta_user_idx]

maxBeta = float(beta_user[max_beta_user_idx])
minBeta = float(beta_user[min_beta_user_idx])


# In[294]:


print(maxUser, minUser, maxBeta, minBeta)
print(type(maxUser), type(minUser), type(maxBeta), type(minBeta))


# In[295]:


answers['Q7'] = [maxUser, minUser, maxBeta, minBeta]


# In[296]:


assert [type(x) for x in answers['Q7']] == [str, str, float, float]


# In[297]:


### Question 8


# In[298]:


est_lambda = None
best_mse = float('inf')

# 测试不同的λ值
lambda_values = numpy.arange(0.1, 10.1, 0.5)

for lambda_reg in tqdm(lambda_values):
    validMSE, _, _, _ = fit_bias_model_sklearn_sparse(ratingsTrain, ratingsValid, lambda_reg)
    if validMSE < best_mse:
        best_mse = validMSE
        best_lambda = lambda_reg

lamb = best_lambda
validMSE = best_mse


# In[299]:


answers['Q8'] = (lamb, validMSE)


# In[300]:


assertFloat(answers['Q8'][0])
assertFloat(answers['Q8'][1])


# In[301]:


predictions = open("predictions_Rating.csv", 'w')
for l in open("pairs_Rating.csv"):
    if l.startswith("userID"): # header
        predictions.write(l)
        continue
    u,b = l.strip().split(',') # Read the user and item from the "pairs" file and write out your prediction
    # (etc.)
    
predictions.close()


# In[354]:


f = open("answers_hw3.txt", 'w')
f.write(str(answers) + '\n')
f.close()


# In[ ]:




