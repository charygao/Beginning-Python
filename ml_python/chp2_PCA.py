#!/usr/bin/env python
# -*-encoding:utf-8-*-

#section1: 线性相关矩阵的秩计算
import numpy as np
M =  np.array([[1,2],[2,4]])
print '计算矩阵秩：',np.linalg.matrix_rank(M,tol=None)


#section2: 手写数字图片经PCA压缩后的二维空间分布
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

from sklearn.decomposition import PCA
estimator = PCA(n_components=2)
x_pca = estimator.fit_transform(x_train)

def plot_pca_scatter():
    colors = ['black','blue','purple','yellow','white','red','lime','cyan','orange','gray']
    for i in range(len(colors)):
        px=x_pca[:,0][y_train.as_matrix()==i]
        py=x_pca[:,1][y_train.as_matrix()==i]
        plt.scatter(px,py,c=colors[i])
    plt.legend(np.arange(0,10).astype(str))
    plt.xlabel('first principle component')
    plt.ylabel('second principle component')
    plt.show()
    print ''

plot_pca_scatter()


#section3: 使用原始64维特征和PCA压缩后的20维特征，在相同配置的支持向量机（分类）模型上分别进行图像识别
from sklearn.svm import LinearSVC

#64维
svc = LinearSVC()
svc.fit(x_train,y_train)
y_predict = svc.predict(x_test)

#20维
estimator = PCA(n_components=20)
pca_x_train = estimator.fit_transform(x_train)
pca_x_test = estimator.transform(x_test)

pca_svc = LinearSVC()
pca_svc.fit(pca_x_train,y_train)
pca_y_predict = pca_svc.predict(pca_x_test)

#性能对比
from sklearn.metrics import classification_report

print '64维特征时：',svc.score(x_test,y_test)
print classification_report(y_test,y_predict,target_names=np.arange(10).astype(str))

print '20维特征时：',pca_svc.score(pca_x_test,y_test)
print classification_report(y_test,pca_y_predict,target_names=np.arange(10).astype(str))


        