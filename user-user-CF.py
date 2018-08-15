import time
import numpy as np
import sys
from pyspark import SparkContext

sc = SparkContext(appName="UserBasedCF")

def getPearsonCC(X,Y):
    mask = np.array(X) + np.array(Y)
    mask = mask / mask
    X = X * mask
    Y = Y * mask
    N = float(np.nansum((X - np.nanmean(X))*(Y - np.nanmean(Y))))
    D = np.sqrt(np.nansum(np.square((X - np.nanmean(X))))) * np.sqrt(np.nansum(np.square((Y - np.nanmean(Y)))))
    if D == 0:
        return 0
    return(N/D)

def CF(test):
    prediction = []
    for row in test:
        PCC = {}
        userId = row[0][0]
        itemId = row[0][1]
        actual = row[1]
        
        if itemId not in train_items:
            prediction.append(((userId,itemId),(actual,np.nanmean(char_matrix[userId]))))
            continue
        #users = map(lambda x: x[0][0], filter(lambda x: x[0][1] == itemId,train))
        users = movies2[itemId]
    
        for user in users:
            coef = getPearsonCC(char_matrix[userId],char_matrix[user])
            if coef != 0:
                PCC[user] = coef
        
        r_dash = np.nanmean(char_matrix[userId])
        
        if len(PCC) == 0:
            prediction.append(((userId,itemId),(actual,r_dash)))
            continue

        mask = np.array(char_matrix[userId]) / np.array(char_matrix[userId])
        
        R = 0
        W = 0
        for user,pcc in PCC.items():
            r = train_kv[(user,itemId)]
            r_avg = np.nanmean(np.array(char_matrix[user]) * mask)
            R += (r - r_avg) * pcc
            W += np.abs(pcc)
        
        pred = r_dash + float(R)/W
        prediction.append(((userId,itemId),(actual,pred)))
    
    return(prediction)

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
    
start = time.time()

input_file = sys.argv[1]
testing_file = sys.argv[2] 

output_file = 'Nikhit_Mago_UserBasedCF.txt'

data = sc.textFile(input_file)
header = data.first()
data = data.filter(lambda x: x!=header).map(lambda x: x.split(',')).map(lambda x: ((int(x[0]), int(x[1])), float(x[2])))
data.persist()

data_test = sc.textFile(testing_file)
header = data_test.first()
data_test = data_test.filter(lambda x: x!=header).map(lambda x: x.split(',')).map(lambda x: ((int(x[0]), int(x[1])), float(-1.0)))
data_test.persist()

train = data.subtractByKey(data_test)

test = data.subtractByKey(train)

data.unpersist()
data_test.unpersist()

train.persist()
test.persist()

users = train.map(lambda x: (x[0][0],1)).groupByKey().sortByKey().map(lambda x: x[0]).collect()
users_len = len(users)

movies = train.map(lambda ((x,y),z): (y,(x,z))).groupByKey().map(lambda x: (x[0],list(set(x[1])))).sortByKey().collect()
movies_len = len(movies)

char_matrix = {k:[] for k in users}
for user in users:
    for movie in movies:
        flag = 0
        for m in movie[1]:
            if user == m[0]:
                flag = 1
                break
        if flag == 1:
            char_matrix[user].append(m[1])
        else:
            char_matrix[user].append(np.NaN)

movies1 = train.map(lambda x: (int(x[0][1]),int(x[0][0]))).groupByKey().map(lambda x: (x[0],list(set(x[1])))).sortByKey().collect()

movies2 = {}
for movie in movies1:
    movies2[movie[0]] = movie[1]

train = train.collect()

train_items = np.unique([t[0][1] for t in train])

train_kv = {}
for i in train:
    train_kv[i[0]] = i[1]

b = test.repartition(200).mapPartitions(CF).collect()

eval_matrix = sc.parallelize(b,4)

rmse = np.sqrt(eval_matrix.map(lambda x: (x[1][0] - x[1][1])**2).mean())

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
