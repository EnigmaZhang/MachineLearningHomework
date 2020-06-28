import numpy as np
import pandas as pd
from pandas.plotting import radviz
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.decomposition import FastICA

# Dimension we need.
P = 2
data = []

with open(r"./data/iris.data", mode="r") as f:
    for lines in f.readlines():
        data.append(lines.strip().split(","))

# A [""] as last
data = data[:-1]
trans_dict = {"Iris-setosa": "0", "Iris-versicolor": "1", "Iris-virginica": "2"}
for i in data:
    i[-1] = trans_dict[i[-1]]

data = np.asarray(data, dtype="float32")
numpy_data = data.copy()
data = pd.DataFrame(data)
# Change columns to string.
data.rename(mapper=lambda x: str(x), axis=1, inplace=True)
radviz(data, class_column="4")
plt.show()

x, y = np.split(numpy_data, (4, ), axis=1)

# PCA
# Col as a var, row as attr.
M = np.mean(x, axis=0)
x = x - M

C = np.cov(x.T)

eigenvalue, eigenvetcor = np.linalg.eig(C)
# Get the index of reversed sorted eigenvalue
sorted_index = np.argsort(-eigenvalue)[:P]
# Get the first p vectors to get the projection matrix.

Projection = eigenvetcor[:, sorted_index]
N = np.dot(x, Projection)

plt.scatter(N[:, 0], N[:, 1], c=y)
plt.show()

data = np.concatenate([N, y], axis=1)
data = pd.DataFrame(data)
# Change columns to string.
data.rename(mapper=lambda x: str(x), axis=1, inplace=True)
radviz(data, class_column="{}".format(P))
plt.show()

# LDA
x, y = np.split(numpy_data, (4, ), axis=1)
origin_y = y.copy()
y = y.squeeze(1)
lda = LinearDiscriminantAnalysis(n_components=P)
lda.fit(x, y)
x = lda.transform(x)
plt.scatter(x[:, 0], x[:, 1], c=y)
plt.show()
data = np.concatenate([x, origin_y], axis=1)
data = pd.DataFrame(data)
# Change columns to string.
data.rename(mapper=lambda x: str(x), axis=1, inplace=True)
radviz(data, class_column="{}".format(P))
plt.show()

# ICA
x, y = np.split(numpy_data, (4, ), axis=1)
y = y.squeeze(1)
ica = FastICA(n_components=P)
ica.fit(x, y)
x = ica.transform(x)
plt.scatter(x[:, 0], x[:, 1], c=y)
plt.show()
data = np.concatenate([x, origin_y], axis=1)
data = pd.DataFrame(data)
# Change columns to string.
data.rename(mapper=lambda x: str(x), axis=1, inplace=True)
radviz(data, class_column="{}".format(P))
plt.show()
