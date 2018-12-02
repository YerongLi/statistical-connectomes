from scipy import io
import matplotlib.pyplot as plt
import sys
from mpl_toolkits.mplot3d import Axes3D
import time
import numpy as np
from sklearn.cluster import KMeans
import pickle
import h5py
import numpy as np
from matplotlib.colors import ListedColormap, BoundaryNorm    
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Line3DCollection
default_timer = time.time
import colorsys
 
def make_colors (num):
    '''Generate num disting colors as RGB triples'''
    HSV_tuples = [(x*1.0/num, 0.5, 0.5) for x in range(num)]
    return list(map(lambda x: colorsys.hsv_to_rgb(*x), HSV_tuples))

filepath = '../DATA/meng_data_10_9_18.mat'
# Number of clusters
n_clusters = 2
# person id
id = 200

def get_fibers(id):
	f = h5py.File(filepath)
	data = f.get('all_endpoints')

	startPoints = f.get(data[id][0])[0][0]; startPoints = np.array(f.get(startPoints)[:]).T
	endPoints = f.get(data[id][0])[1][0]; endPoints = np.array(f.get(endPoints)[:]).T

	X= np.concatenate((startPoints, endPoints), axis =1)#[1:10]
	return X
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
X = get_fibers(id)
segments= X.reshape(-1,2,3)

with open('140KMeans200:201.pkl', 'rb') as f:
	_, _, labels = pickle.load(f)
print(len(labels))
n_clusters = len(labels)
# kmeans = KMeans(n_clusters=n_clusters)
# Fitting the input data
# kmeans = kmeans.fit(X)
# Getting the cluster labels
# labels = kmeans.predict(X)

color_list = make_colors(30)
#print(color_list)
#color_list = ['r', 'b', 'g']
for i in range(30):
	lc = Line3DCollection(segments[labels==i], colors= color_list[i])
	ax.add_collection3d(lc)

ax.set_xlabel('X')
ax.set_xlim3d(0, 175)
ax.set_ylabel('Y')
ax.set_ylim3d(-1, 200)
ax.set_zlabel('Z')
ax.set_zlim3d(0, 175)
plt.show()
#print(startPoints.shape, endPoints.shape)

