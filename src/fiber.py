import matplotlib.pyplot as plt
import sys
from mpl_toolkits.mplot3d import Axes3D
import time
import numpy as np
from sklearn.cluster import KMeans

import h5py
from matplotlib.colors import ListedColormap, BoundaryNorm    
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Line3DCollection

import colorsys
import pickle
import multiprocessing

# Number of clusters
n_clusters = 140
# number of people 
# 0 -1064

ids = range(0, 1064 +1)
print(ids)
# threading
n_processes = 2 
filepath = '../DATA/meng_data_10_9_18.mat'

f = h5py.File(filepath, 'r', libver='latest', swmr=True)

data = f.get('all_endpoints')
n_people = 1065
startPoints_data = [np.array(f.get(f.get(data[id][0])[0][0])[:]).T for id in range(n_people)]
endPoints_data = [np.array(f.get(f.get(data[id][0])[1][0])[:]).T for id in range(n_people)]

startPoints_data = startPoints_data[200:300]
endPoints_data = endPoints_data[200:300]
def make_colors (num):
	'''Generate num disting colors as RGB triples'''
	HSV_tuples = [(x*1.0/num, 0.5, 0.5) for x in range(num)]
	return list(map(lambda x: colorsys.hsv_to_rgb(*x), HSV_tuples))


'''
'''
startPoints = np.vstack(startPoints_data)
endPoints = np.vstack(endPoints_data)
del startPoints_data, endPoints_data
X= np.concatenate((startPoints, endPoints), axis =1)#[1:10]
del startPoints, endPoints
kmeans = KMeans(n_clusters=n_clusters)
# Fitting the input data
kmeans = kmeans.fit(X)
labels = kmeans.predict(X)
with open(str(n_clusters)+'KMeans200-299.pkl', 'wb') as f:
	pickle.dump(labels, f)
'''
t0 = time.time()
pool = multiprocessing.Pool(processes=n_processes)
labels = pool.map_async(cluster, ids)
t1 = time.time()

labels = labels.get()
pool.close();
pool.join();
print(t1-t0)

with open(str(n_clusters)+'labels'+str(min(ids))+'-'+str(max(ids))+'.pkl', 'wb') as f:
	pickle.dump(labels, f)

with open(str(n_clusters)+'labels'+str(min(ids))+'-'+str(max(ids))+'.pkl', 'rb') as f:
	print(pickle.load( f))
'''
'''
segments= X.reshape(-1,2,3)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
color_list = make_colors(n_clusters)
for i in range(n_clusters):
	lc = Line3DCollection(segments[labels==i], colors= color_list[i])
	ax.add_collection3d(lc)

ax.set_xlabel('X')
ax.set_xlim3d(0, 175)
ax.set_ylabel('Y')
ax.set_ylim3d(-1, 200)
ax.set_zlabel('Z')
ax.set_zlim3d(0, 175)'''

#kmeans = KMeans(n_clusters=5, random_state=0).fit(X)
#index = (kmeans.labels_==1)
#print(idex)

'''plt.show()'''

