import numpy as np
from sklearn.ensemble import IsolationForest
import pandas as pd
import matplotlib.pyplot as plt

def shingle(data, shingle_size):
    if np.ndim(data) ==2:
        data = data[:,0]
    num_data = len(data)
    shingled_data = np.zeros((num_data - shingle_size, shingle_size))

    for n in range(num_data - shingle_size):
        shingled_data[n] = data[n : (n + shingle_size)]
    return shingled_data

use_shingle = False
shingle_size = 10

data_path = "data/EQUIPMENT_029/2022-09-24_00:00:00.csv"
df = pd.read_csv(data_path, index_col=0)
X = df.values
X_val = X.copy()

##X[100,0] = 5000.
##X = (X[:,0]).reshape(-1,1)
#X = np.sin(np.linspace(0,np.pi * 8 , 1000)).reshape(-1,1) + 2.
##X = np.ones(1000).reshape(-1,1) + np.random.rand(1000)
#
#
#X_val = X.copy()
#X_val[100:110,0] = np.random.rand(10) +1.

if use_shingle:
    X = shingle(X, shingle_size)
    X_val = shingle(X_val, shingle_size)

print("training IsolationForest")
clf = IsolationForest(random_state=0).fit(X)
anomaly = clf.score_samples(X_val)

fig, ax = plt.subplots(3,1)
ax[0].plot(X_val[:,0])
ax[0].set_title("raw data")
ax[1].plot(anomaly*(-0.5) + 0.5)
ax[1].set_title("Anomary score: Isoration forest")



import numpy as np
import rrcf

# Generate data

# Set tree parameters
num_trees = 50
tree_size = 256

# Create a forest of empty trees
forest = []
for _ in range(num_trees):
    tree = rrcf.RCTree()
    forest.append(tree)
    
points = X_val.copy()

# Create a dict to store anomaly score of each point
avg_codisp = {}

print("training Robust Random Cut Forest")
# For each shingle...
for index, point in enumerate(points):
    # For each tree in the forest...
    for tree in forest:
        # If tree is above permitted size, drop the oldest point (FIFO)
        if len(tree.leaves) > tree_size:
            tree.forget_point(index - tree_size)
        # Insert the new point into the tree
        tree.insert_point(point, index=index)
        # Compute codisp on the new point and take the average among all trees
        if not index in avg_codisp:
            avg_codisp[index] = 0
        avg_codisp[index] += tree.codisp(index) / num_trees

ax[2].plot(avg_codisp.values())
ax[2].set_title("Anomaly score: Robust Random Cut Forest")
plt.show()
