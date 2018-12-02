from sklearn import linear_model
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LassoCV, RidgeCV
import pandas as pd
import time, sys, h5py, pickle
import numpy as np

try:
  n_clusters = int(sys.argv[1])
except:
  print('Please input number of clusters')
  sys.exit(1);

df = pd.read_csv('../DATA/trait_labels.csv')
content = [list(df.Subject), list(df.PicVocab_AgeAdj), list(df.ReadEng_AgeAdj)] #you can also use df['column_name']
PicV = {content[0][i]: content[1][i] for i in range(len(content[0]))}
Read = {content[0][i]: content[2][i] for i in range(len(content[0]))}


filepath = '../DATA/meng_data_10_9_18.mat'

f = h5py.File(filepath, 'r', libver='latest', swmr=True)
data = f.get('allsub')
n_people = 1065
subjects = data[0].astype(int)

PicV_labels = np.array([PicV[sub] for sub in subjects])
Read_labels = np.array([Read[sub] for sub in subjects])

with open(str(n_clusters)+'freq.pkl', 'rb') as lbl:
    X = pickle.load(lbl)
X = X[:, :-1]

y= PicV_labels
X= np.concatenate((X,y.reshape((-1,1))), axis=1); del y

y = Read_labels
X= np.concatenate((X,y.reshape((-1,1))), axis=1); del y
#print(X.shape)
#X = X[:,:-1]
start = 565
tol = 1e-4
def avg_PicV(X):
        np.random.shuffle(X)
        reg = LassoCV(cv=5, random_state=0, tol = tol)
        reg.fit(X[:start, :-2], X[:start, -2].reshape((-1,)))
        y_ = reg.predict(X[start:, :-2])
        return mean_squared_error(X[start:, -2], y_)
def avg_Read(X):
        np.random.shuffle(X)
        #reg = LassoCV(cv=5, random_state=0)
        reg = LassoCV(cv=5, random_state=0, tol = tol)
        reg.fit(X[:start, :-2], X[:start, -1].reshape((-1,)))
        y_ = reg.predict(X[start:, :-2])
        return mean_squared_error(X[start:, -1], y_)

avgPicV = np.mean([avg_PicV(X) for i in range(100)])
avgRead = np.mean([avg_Read(X) for i in range(100)])
print('number of clusters:',n_clusters)
print(avgPicV)
print(avgRead)
