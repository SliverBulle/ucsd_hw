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
import tqdm
def readGz(path):
    for l in gzip.open(path, 'rt'):
        yield eval(l)

def readCSV(path):
    f = gzip.open(path, 'rt')
    f.readline()
    for l in f:
        u,b,r = l.strip().split(',')
        r = int(r)
        yield u,b,r
allRatings = []
for l in readCSV("train_Interactions.csv.gz"):
    allRatings.append(l)
ratingsTrain = allRatings[:190000]
ratingsValid = allRatings[190000:]
ratingsPerUser = defaultdict(list)
ratingsPerItem = defaultdict(list)
userPerItem = defaultdict(set)
itemPerUser = defaultdict(set)
for u,b,r in allRatings:
    ratingsPerUser[u].append((b,r))
    ratingsPerItem[b].append((u,r))
    userPerItem[u].add(b)
    itemPerUser[b].add(u)

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

def sample_negative(user, positive_books, all_books, num_samples=1):
    negative_samples = []
    while len(negative_samples) < num_samples:
        neg_book = random.choice(all_books)
        if neg_book not in positive_books:
            negative_samples.append(neg_book)
    return negative_samples


validation_positive = ratingsValid
validation_negative = []
all_books = list(bookCount.keys())

user_positive_books = defaultdict(set)
for u, b, r in ratingsTrain:
    user_positive_books[u].add(b)

for u, b, r in validation_positive:
    negative_books = sample_negative(u, user_positive_books[u], all_books)
    for neg_b in negative_books:
        validation_negative.append((u, neg_b, 0))


validation_set = [(u, b, 1) for u, b, r in validation_positive] + validation_negative



def jaccard_similarity(s1, s2):
    intersection = s1 & s2
    union = s1 | s2
    if not union:
        return 0
    return len(intersection) / len(union)

#read predictions_Read.csv
return1 = set()
count = 0
for ic, i in mostPopular:
    count += ic
    return1.add(i)
    if count > totalRead * 0.7:
        break
with open("predictions_Read.csv", 'w') as predictions:
    for l in open("pairs_Read.csv"):
        if l.startswith("userID"):
            predictions.write(l)
            continue
        u, b = l.strip().split(',')
        
        popularity_pred = 1 if b in return1 else 0
        read_items = [item for item, r in ratingsPerUser[u]]
        similarities = [jaccard_similarity(userPerItem[b], userPerItem[b_prime]) for b_prime in read_items]
        #a book is likely purchased if the book is similar to any of the books the user has purchased
        #a user is likely to purchase a book when the user is similar to any of the users who have purchased the book
        if similarities:
            jaccard_pred = 1 if max(similarities) > 0.112 else 0
        else:
            users = [ user for user, r in ratingsPerItem[b]]
            similarities = [jaccard_similarity(itemPerUser[u], itemPerUser[user]) for user in users]
            if similarities:    
                jaccard_pred = 1 if max(similarities) > 0.112 else 0
            else:
                jaccard_pred = 0
        prediction = 1 if max(popularity_pred, jaccard_pred) else 0
        predictions.write(f"{u},{b},{prediction}\n")

#rating predictions

from surprise import Dataset, Reader, SVD
from surprise.model_selection import train_test_split, GridSearchCV
from surprise import accuracy
import pandas as pd

# 加载数据到 DataFrame
df = pd.DataFrame(allRatings, columns=['userId', 'bookId', 'rating'])

# 定义 Reader
reader = Reader(rating_scale=(1, 5))

# 加载数据到 Surprise 的 Dataset
data = Dataset.load_from_df(df[['userId', 'bookId', 'rating']], reader)

model = SVD(
    n_epochs=20,
    lr_all=0.007,
    reg_all=0.1,
    n_factors=20
)
full_trainset = data.build_full_trainset()
# 训练模型
model.fit(full_trainset)
# in[49] 生成 predictions_Rating_SVD.csv
with open("predictions_Rating.csv", 'w') as predictions:
    for l in open("pairs_Rating.csv"):
        if l.startswith("userID"):
            predictions.write(l)
            continue
        u, b = l.strip().split(',')
        pred = model.predict(u, b).est

        predictions.write(f"{u},{b},{pred}\n")


