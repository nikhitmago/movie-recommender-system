import time
import numpy as np
from pyspark.mllib.recommendation import ALS, MatrixFactorizationModel, Rating
import sys
from pyspark import SparkContext

sc = SparkContext(appName="ModelBasedCF")

# In[2]:


start = time.time()


# In[3]:


input_file = sys.argv[1]
testing_file = sys.argv[2]


# In[4]:


if testing_file.find('testing_small') != -1:
    output_file = 'Nikhit_Mago_ModelBasedCF_Small.txt'
else:
    output_file = 'Nikhit_Mago_ModelBasedCF_Big.txt'


# In[5]:


data = sc.textFile(input_file)
header = data.first()
data = data.filter(lambda x: x!=header).map(lambda x: x.split(',')).map(lambda x: ((int(x[0]), int(x[1])), float(x[2])))
data.persist()


# In[6]:


data_test = sc.textFile(testing_file)
header = data_test.first()
data_test = data_test.filter(lambda x: x!=header).map(lambda x: x.split(',')).map(lambda x: ((int(x[0]), int(x[1])), float(-1.0)))
data_test.persist()


# In[7]:


train = data.subtractByKey(data_test)


# In[8]:


test = data.subtractByKey(train)


# In[9]:


data.unpersist()
data_test.unpersist()


# In[10]:


train = train.map(lambda ((x,y),z): Rating(x,y,z))
test = test.map(lambda ((x,y),z): Rating(x,y,z))


# In[11]:


train.persist()
test.persist()


# In[12]:


model = ALS.train(train,rank=5,iterations=10,lambda_=0.1)


# In[13]:


predictions = model.predictAll(test.map(lambda x: (x[0],x[1]))).map(lambda x: ((x[0], x[1]), x[2]))


# In[ ]:


eval_matrix = test.map(lambda x: ((x[0], x[1]), x[2])).join(predictions)


# In[ ]:


rmse = np.sqrt(eval_matrix.map(lambda x: (x[1][0] - x[1][1])**2).mean())


# ### MapPartitions way

# In[65]:


def get_abs_errors(errors):
    errors = list(errors)
    l = []
    for error in errors:
        if error >=0 and error <1:
            l.append(('01',1))
        elif error >=1 and error <2:
            l.append(('12',1))
        elif error >=2 and error <3:
            l.append(('23',1))
        elif error >=3 and error <4:
            l.append(('34',1))
        else:
            l.append(('4',1))
    return l


# In[66]:


abs_errors = eval_matrix.map(lambda x: np.abs(x[1][0] - x[1][1])).mapPartitions(get_abs_errors)


# ### Map way

# In[67]:


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


# In[68]:


abs_errors = eval_matrix.map(lambda x: np.abs(x[1][0] - x[1][1])).map(get_abs_errors)


# In[69]:


#skip this line


# In[70]:


levels = abs_errors.reduceByKey(lambda x,y: x+y).sortByKey().collect()


# In[71]:


preds = predictions.sortByKey().map(lambda x: str((x[0][0],x[0][1],x[1]))[1:-1]).collect()


# In[72]:


with open(output_file,'wb') as file_write:
    for pred in preds:
        file_write.write(pred)
        file_write.write('\n')
file_write.close()     


# In[73]:


for level in levels:
    if len(level[0])>1:
        print('>={} and <{}: {}'.format(level[0][0],level[0][1],level[1]))
    else:
        print('>={}: {}'.format(level[0],level[1]))
        
time_taken = str(int(time.time() - start)) + " sec"

print('RMSE: {}'.format(rmse))
print('Time: {}'.format(time_taken))

