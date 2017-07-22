#!/usr/bin/env python
# -*-encoding:utf-8-*-

import pandas as pd 
titanic = pd.read_csv('http://biostat.mc.vanderbilt.edu/wiki/pub/Main/DataSets/titanic.txt')
y = titanic['survived']
x= titanic.drop(['row.names','name','survived'],axis=1)

#对缺失的数据填充
x['age'].fillna(x['age'].mean(),inplace=True)
x.fillna('UNKNOWN',inplace=True)

#数据集分割
from sklearn.cross_validation import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.25,random_state=33)

#类别型特征向量化
from sklearn.feature_extraction import DictVectorizer
vec  = DictVectorizer()
x_train = vec.fit_transform(x_train.to_dict(orient='record'))
x_test = vec.transform(x_test.to_dict(orient='record'))

print '特征向量的维度：',len(vec.feature_names_)

#基于全部特征，使用决策树模型进行分类预测
from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier(criterion="entropy")
dt.fit(x_train,y_train)
print '基于全部特征时，模型性能评估得分：',dt.score(x_test,y_test)

#筛选前20%的特征，使用决策树模型进行分类预测
from sklearn import feature_selection
fs = feature_selection.SelectPercentile(feature_selection.chi2,percentile=20)
x_train_fs = fs.fit_transform(x_train,y_train)
x_test_fs = fs.transform(x_test)

dt_fs = DecisionTreeClassifier(criterion="entropy")
dt_fs.fit(x_train_fs,y_train)
print '基于前20%特征时，模型性能评估得分：',dt_fs.score(x_test_fs,y_test)
print ''


#通过交叉验证的方法，按照固定的间隔百分比筛选特征，并作图展示性能随特征筛选比例的变化
from sklearn.cross_validation import cross_val_score
import numpy

percentiles = range(1,100,2)
results = []

for i in percentiles:
    fs = feature_selection.SelectPercentile(feature_selection.chi2,percentile=i)
    x_train_fs = fs.fit_transform(x_train,y_train)
    scores = cross_val_score(dt,x_train_fs,y_train,cv=5)
    results = numpy.append(results,scores.mean())
print results

#找到最佳的性能的特征筛选比
opt = numpy.where(results==results.max())
print 'Optimal number of feature is %d percent ' %percentiles[opt[0][0]]
print ''


import pylab as pl 
pl.plot(percentiles,results)
pl.xlabel('percentile of features ')
pl.ylabel('accuracy')
pl.show()

#最佳特征比时的模型性能
fs = feature_selection.SelectPercentile(feature_selection.chi2,percentile=7)
x_train_fs = fs.fit_transform(x_train,y_train)
x_test_fs = fs.transform(x_test)
dt.fit(x_train_fs,y_train)

print "基于前7%特征时，模型性能评估得分：",dt.score(x_test_fs,y_test)
print ''










