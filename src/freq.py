import matplotlib.pyplot as plt
import sys
import collections
from mpl_toolkits.mplot3d import Axes3D
import time
import numpy as np
from sklearn.cluster import MiniBatchKMeans

import h5py
from matplotlib.colors import ListedColormap, BoundaryNorm    
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Line3DCollection

import colorsys
import pickle
import multiprocessing
# Number of clusters
try:
  n_clusters = int(sys.argv[1])
except:
  print('Please input number of clusters')
  sys.exit(1);
# number of people 
# 0 -1064
with open('../DATA/'+str(n_clusters)+'KMeans.pkl', 'rb') as f:
  _,_,labels = pickle.load(f)

with open('../DATA/length.pkl', 'rb') as f:
  length = pickle.load(f)

acc_labels = []
tmp = labels[:]
for i in range(len(length)-1): 
  try:
    acc_labels.append(tmp[:length[i]])
    tmp = tmp[length[i]:]
  except:
    print(i)
    break;

acc_labels.append(tmp)

counts = [collections.Counter(sub) for sub in acc_labels]
counts = np.array([[ct[i] for i in range(n_clusters)] for ct in counts])

row_sums = counts.sum(axis=1)

freq = counts/ row_sums[:, np.newaxis]

with open(str(n_clusters)+'freq.pkl', 'wb') as lbl:
    pickle.dump((freq), lbl)
