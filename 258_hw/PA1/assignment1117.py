import gzip
from collections import defaultdict
import numpy
from sklearn.preprocessing import normalize
import tqdm
from sklearn import svm
def readGz(path):
  for l in gzip.open(path, 'rt'):
    yield eval(l)

def readCSV(path):
  f = gzip.open(path, 'rt')
  f.readline()
  for l in f:
    yield l.strip().split(',')

### Rating baseline: compute averages for each user, or return the global average if we've never seen the user before
import time
start_time = time.time()
print("ratings start")
allRatings = []
userRatings = defaultdict(list)


allRatings = []
for l in readCSV("train_Interactions.csv.gz"):
    allRatings.append(l)

ratingsTrain = allRatings[:190000]
ratingsValid = allRatings[190000:]
ratingsPerUser = defaultdict(list)
ratingsPerItem = defaultdict(list)
usersPerItem = defaultdict(set)
itemsPerUser = defaultdict(set)

#print(ratingsTrain[0])

for u,b,r in ratingsTrain:
    ratingsPerUser[u].append((b,int(r)))
    ratingsPerItem[b].append((u,int(r)))
    usersPerItem[b].add(u)
    itemsPerUser[u].add(b)

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
    if count > totalRead*0.5:
        break

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
    
    return valid_mse, model.intercept_, model.coef_[:n_users], model.coef_[n_users:], model

# 运行模型
#validMSE, alpha, beta_user, beta_item, model = fit_bias_model_sklearn_sparse(ratingsTrain, ratingsValid)

lambda_reg = 4.6

from sklearn.preprocessing import OneHotEncoder
encoder = OneHotEncoder()
user_book_features = encoder.fit_transform([[u,b] for u,b,_ in ratingsTrain])

X_train = user_book_features
y_train = numpy.array([r for _,_,r in ratingsTrain])

model = svm.LinearSVC(class_weight='balanced')
model.fit(X_train, y_train)

users = list(set(u for u,_,_ in ratingsTrain))
items = list(set(b for _,b,_ in ratingsTrain))

user_to_idx = {u:i for i,u in enumerate(users)}
item_to_idx = {b:i for i,b in enumerate(items)}
predictions = open("predictions_Rating.csv", 'w')
print("start predicting")
for l in open("pairs_Rating.csv"):
    if l.startswith("userID"):
        predictions.write(l)
        continue
    u,b = l.strip().split(',')
    
    try:
        # 使用与训练时相同的encoder进行转换
        feature_vector = encoder.transform([[u,b]])
        prediction = model.predict(feature_vector)[0]
    except:
        # 处理未知用户或书籍的情况
        prediction = 3  # 使用默认评分
    
    # 确保评分在合理范围内

    predictions.write(u + ',' + b + ',' + str(int(prediction)) + '\n')

predictions.close()




### Would-read baseline: just rank which books are popular and which are not, and return '1' if a book is among the top-ranked
print("start time at:", time.time()-start_time)
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD
import numpy as np
from collections import defaultdict
from tqdm import tqdm
from sklearn.preprocessing import normalize
from sklearn import svm

def Jaccard(s1, s2):
    upper = len(s1 & s2)
    lower = len(s1 | s2)
    return upper / lower
X_train = []
y_train = []

for u, b, r in ratingsTrain:
    # 构建特征向量
    feature_vector = [1 if b in return1 else 0]  # 特征1：物品是否在最受欢迎的集合中
    user_books = itemsPerUser[u]
    max_sim = max([Jaccard(usersPerItem[b], usersPerItem[read_book]) for read_book in user_books], default=0)
    feature_vector.append(1 if max_sim > 0.01 else 0)  # 特征2：最大Jaccard相似度是否超过阈值
    X_train.append(feature_vector)
    y_train.append(1 if int(r) > 0 else 0)  # 标签：如果评分大于0，则为1，否则为0

# 使用模型
model = svm.LinearSVC()
model.fit(X_train, y_train)


print("生成预测结果...")
predictions = open("predictions_Read.csv", 'w')
predictions.write("userID,bookID,prediction\n")

# 打开预测结果文件
predictions = open("predictions_Read.csv", 'w')
batch_size = 1000
current_batch = []
# 读取并处理测试集
with open("pairs_Read.csv", 'r') as test_file:
    for l in test_file:
        if l.startswith("userID"):
            predictions.write(l)  # 写入表头
            continue
        u, b = l.strip().split(',')
        if len(current_batch) >= batch_size:
            # 批量预测
            preds = model.predict(current_batch)
            # 写入预测结果
            for (user, book), pred in zip(current_batch, preds):
                binary_pred = 1 if pred > 0.5 else 0
                predictions.write(f"{user},{book},{binary_pred}\n")
            current_batch = []
    
        # 处理最后一个批次
        if current_batch:
            preds = model.predict(current_batch)
            for (user, book), pred in zip(current_batch, preds):
                binary_pred = 1 if pred > 0.5 else 0
                predictions.write(f"{user},{book},{binary_pred}\n")

# 关闭预测结果文件
predictions.close()

