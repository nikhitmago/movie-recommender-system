import numpy as np
import random
import time
import sys
from itertools import combinations
import sys
import random
from pyspark import SparkContext

sc = SparkContext(appName="ItemBasedCF")

start = time.time()

input_file = sys.argv[1]
testing_file = sys.argv[2]

output_file = 'Nikhit_Mago_ItemBasedCF.txt'

### LSH

hashLen = 60
numBands = 20
data = sc.textFile(input_file)
header = data.first()
users = data.filter(lambda x: x!=header).map(lambda x: x.split(',')).map(lambda x: (int(x[0])-1,int(x[1]))).groupByKey().map(lambda x: (x[0],list(set(x[1])))).sortByKey().collect()
users_len = len(users)

movies = data.filter(lambda x: x!=header).map(lambda x: x.split(',')).map(lambda x: (int(x[1]),1)).groupByKey().sortByKey().map(lambda x: x[0]).collect()
movies_len = len(movies)

char_matrix = {}
for movie in movies:
    char_matrix[movie] = [1 if movie in user[1] else 0 for user in users]

def hashFunc(x,a,b,m,i):
    return map(lambda x: (a*x+b)%m ,x)

hash_fns = []
for i in range(hashLen): 
    hash_fns.append(hashFunc(range(users_len),random.randint(0,671),random.randint(0,671),users_len,i))
hash_fns = np.array(hash_fns)
hash_fns = hash_fns.T

matrix = np.array([char_matrix[movie] for movie in movies]).T

signatures = {movie:np.array([sys.maxint for i in range(len(hash_fns.T))]) for movie in movies}

for i in range(users_len):
    movies_1 = np.array(movies)[np.argwhere(matrix[i] == 1).flatten()]
    for movie in movies_1:
        signatures[movie] = list(np.minimum(signatures[movie],hash_fns[i]))

signatures_arr = np.array([signatures[movie] for movie in movies]).T.tolist()

signatures_arr = sc.parallelize(signatures_arr,numBands)

def getCandidates(rows,movies):
    l = []
    rows = np.array(list(rows))
    
    d = {tuple(rows[:,k]):[] for k in range(rows.shape[1])}
    for i in range(rows.shape[1]):
        d[tuple(rows[:,i])].append(movies[i])
        
    for k,v in d.items():
        if len(v)>1:
            l += list(combinations(sorted(v),2))
    return(l)

def getJaccardSimilarity(candidates):
    l = []
    #candidates = list(candidates)
    for candidate in candidates:
        N = 0
        D = 0
        movie1 = char_matrix[candidate[0]]
        movie2 = char_matrix[candidate[1]]
        for i in range(len(movie1)):
            sum_ = movie1[i] + movie2[i]
            if(sum_ >= 1):
                D += 1
                if(sum_ == 2):
                    N += 1
        score = float(N)/D
        l.append((candidate,score))
    return(l)   

result = signatures_arr.mapPartitions(lambda x: getCandidates(x,movies)).map(lambda x: (x,1)).groupByKey().map(lambda x: x[0]).mapPartitions(getJaccardSimilarity).filter(lambda (x,y): y >= 0.5).collect()

jaccardSim = {k[0]:k[1] for k in result}

data = sc.textFile(input_file)
header = data.first()
data = data.filter(lambda x: x!=header).map(lambda x: x.split(',')).map(lambda x: ((int(x[0]), int(x[1])), float(x[2])))
data.persist()

#len(data.collect())

data_test = sc.textFile(testing_file)
header = data_test.first()
data_test = data_test.filter(lambda x: x!=header).map(lambda x: x.split(',')).map(lambda x: ((int(x[0]), int(x[1])), float(-1.0)))
data_test.persist()

#len(data_test.collect())

train = data.subtractByKey(data_test)

test = data.subtractByKey(train)

# len(test.collect())

data.unpersist()
data_test.unpersist()

train.persist()
test.persist()

users1 = train.map(lambda x: (int(x[0][0]),int(x[0][1]))).groupByKey().map(lambda x: (x[0],list(set(x[1])))).sortByKey().collect()

users2 = {}
for user in users1:
    users2[user[0]] = user[1]

users = train.map(lambda x: (x[0][0],1)).groupByKey().sortByKey().map(lambda x: x[0]).collect()

movies = train.map(lambda ((x,y),z): (y,(x,z))).groupByKey().map(lambda x: (x[0],list(set(x[1])))).sortByKey().collect()

train = train.collect()

train_kv = {}
for i in train:
    train_kv[i[0]] = i[1]

train_items = np.unique([t[0][1] for t in train])

char_matrix_u = {k:[] for k in users}
for user in users:
    for movie in movies:
        flag = 0
        for m in movie[1]:
            if user == m[0]:
                flag = 1
                break
        if flag == 1:
            char_matrix_u[user].append(m[1])
        else:
            char_matrix_u[user].append(np.NaN)

def getJaccardSim(movie1,movie2):
    N = 0
    D = 0
    for i in range(len(movie1)):
        sum_ = movie1[i] + movie2[i]
        if(sum_ >= 1):
            flag = 1
            D += 1
            if(sum_ == 2):
                N += 1
    if D == 0:
        return 0
    return(float(N)/D)

def CF(test):
    prediction = []
    for row in test:
        JS = {}
        userId = row[0][0]
        itemId = row[0][1]
        actual = row[1]
        
        if itemId not in train_items:
            prediction.append(((userId,itemId),(actual,np.nanmean(char_matrix_u[userId]))))
            continue  
        movies = users2[userId]
    
        for movie in movies:
            if tuple(sorted((movie,itemId))) in jaccardSim:
                JS[movie] = jaccardSim[tuple(sorted((movie,itemId)))]
            else:
                JS[movie] = getJaccardSim(char_matrix[itemId],char_matrix[movie])

        
        r_dash = np.nanmean(char_matrix_u[userId])
        
        if len(JS) == 0:
            prediction.append(((userId,itemId),(actual,r_dash)))
            continue
        
        N = len(JS)
        nn = sorted(JS.items(), key = lambda x: x[1])[:N]
        
        R = 0
        W = 0
        for movie,jaccard in nn:
            R += train_kv[(userId,movie)]*jaccard
            W += np.abs(jaccard)
        
        pred = float(R)/W
        prediction.append(((userId,itemId),(actual,pred)))
    
    return(prediction)

b = test.repartition(200).mapPartitions(CF).collect()

eval_matrix = sc.parallelize(b,4)

rmse = np.sqrt(eval_matrix.map(lambda x: (x[1][0] - x[1][1])**2).mean())

def get_abs_errors(error):
    if error >=0 and error <1:
        return(('01',1))
    elif error >=1 and error <2:
        return(('12',1))
    elif error >=2 and error <3:
        return(('23',1))
    elif error >=3 and error <4:
        return(('34',1))
    else:
        return(('4',1))

abs_errors = eval_matrix.map(lambda x: np.abs(x[1][0] - x[1][1])).map(get_abs_errors)

levels = abs_errors.reduceByKey(lambda x,y: x+y).sortByKey().collect()

preds = eval_matrix.sortByKey().map(lambda x: str((x[0][0],x[0][1],x[1][1]))[1:-1]).collect()

with open(output_file,'wb') as file_write:
    for pred in preds:
        file_write.write(pred)
        file_write.write('\n')
file_write.close()     

for level in levels:
    if len(level[0])>1:
        print('>={} and <{}: {}'.format(level[0][0],level[0][1],level[1]))
    else:
        print('>={}: {}'.format(level[0],level[1]))
        
time_taken = str(int(time.time() - start)) + " sec"

print('RMSE: {}'.format(rmse))
print('Time: {}'.format(time_taken))

