# -*- coding: utf-8 -*-
"""
Created on Mon May 28 12:42:37 2018

@author: Ramesh
"""
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler 
%matplotlib inline

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
# loading dataset into Pandas DataFrame
df = pd.read_csv(url
                 , names=['sepal length','sepal width',
                          'petal length','petal width','target'])


featuresfeatures = ['sepal length', 'sepal width', 'petal length', 'petal width']
x = df.loc[:, features].values

y = df.loc[:,['target']].values

x = StandardScaler().fit_transform(x)

pd.DataFrame(data = x, columns = features).head()

pca = PCA(n_components=2)

principalComponents = pca.fit_transform(x)

principalDf = pd.DataFrame(data = principalComponents
             , columns = ['principal component 1', 'principal component 2'])

principalDf.head(5)

df[['target']].head()

finalDf = pd.concat([principalDf, df[['target']]], axis = 1)
finalDf.head(5)

fig = plt.figure(figsize = (8,8))
ax = fig.add_subplot(1,1,1) 
ax.set_xlabel('Principal Component 1', fontsize = 15)
ax.set_ylabel('Principal Component 2', fontsize = 15)
ax.set_title('2 Component PCA', fontsize = 20)


targets = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']
colors = ['r', 'g', 'b']
for target, color in zip(targets,colors):
    indicesToKeep = finalDf['target'] == target
    ax.scatter(finalDf.loc[indicesToKeep, 'principal component 1']
               , finalDf.loc[indicesToKeep, 'principal component 2']
               , c = color
               , s = 50)
ax.legend(targets)
ax.grid()

pca.explained_variance_ratio_



# =============================================================================
# def plot3d(X, title):
# 	'''3d plot first 3 components of a data frame'''
# 	fig = plt.figure(1, figsize=(4, 3))
# 	plt.clf()
# 	ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)
# 	plt.cla()
# 
# 	ax.set_xlim(-1, 1)
# 	ax.set_ylim(-1, 1)
# 	ax.set_zlim(-1, 1)
# 
# 	plt.gca().set_aspect('equal', adjustable='box')
# 	ax.scatter(X[:, 0], X[:, 1], X[:, 2], cmap=plt.cm.spectral,
# 	           edgecolor='k')
# 	plt.title(title)
# 	plt.show()
# 
# plot3d(X, 'Raw Data')
# 
# plot3d(X, 'PCA Transformed Data')
# print('PCA Components (column vectors are the eigen vectors)\n{}'.format(pca.components_))
# print('Explained Variance Ratio\n{}'.format(pca.explained_variance_ratio_))
# print('Relative Variance Ratio\n{}'.format(np.cumsum(pca.explained_variance_ratio_)))
# =============================================================================


