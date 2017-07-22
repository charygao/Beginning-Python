#!/usr/bin/env python
# -*-encoding:utf-8-*-

import numpy
import matplotlib.pyplot as plt
import pandas

digits_train = pandas.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/optdigits/optdigits.tra',header=None)
digits_test = pandas.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/optdigits/optdigits.tes',header=None)

#从训练和测试集合上分离出来64维度的像素特征与1维度的数字目标
x_train = digits_train[numpy.arange(64)]
y_train=digits_train[64]

x_test = digits_test[numpy.arange(64)]
y_test = digits_test[64]

#从sklearn.cluster 中导入 KMeans模型
from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=10)
kmeans.fit(x_train)
y_predict = kmeans.predict(x_test)


#性能评估，使用ARI进行Kmeans聚类性能评估
#被评估的数据样本带有正确的类别信息，使用ARI（Adjusted Rand Index）评估性能
from sklearn import metrics
print 'ARI Score of Kmeans Clustrer is ',metrics.adjusted_rand_score(y_test,y_predict)




#肘部观察法选择合适的聚类数K
import numpy as np
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt

#使用均匀分布函数随机生成三个簇，每个簇周围10个样本数据
cluster1 = np.random.uniform(0.5,1.5,(2,10))
cluster2 = np.random.uniform(5.5,6.5,(2,10))
cluster3 = np.random.uniform(3.0,4.0,(2,10))

#绘制30个样本的分布图
x = np.hstack((cluster1,cluster2,cluster3)).T
plt.scatter(x[:,0],x[:,1])
plt.xlabel('x1')
plt.ylabel('x2')
plt.show() # showdialog，需要关闭后继续运行以下程序


#测试9种不同的k值，每个k对应的聚类质量，并作图
K = range(1,10)
meandistortions = []

for k in K:
    print 'the cluster count is ',k
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(x)
    meandistortions.append(sum(np.min(cdist(x,kmeans.cluster_centers_,'euclidean'),axis=1))/x.shape[0])

plt.plot(K,meandistortions,'bx-')
plt.xlabel('k')
plt.ylabel('average dispersion')
plt.title('Select k with the Elbow Method')
plt.show()


