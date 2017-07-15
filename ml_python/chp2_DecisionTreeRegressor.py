#!/usr/bin/env python
# -*-encoding:utf-8-*-


#数据
from sklearn.datasets import load_boston
boston = load_boston()
print boston.DESCR

#数据集合分割
from sklearn.cross_validation import train_test_split
import numpy
x= boston.data
y= boston.target
x_train,x_test,y_train,y_test = train_test_split(x,y,random_state=33,test_size=0.25)

#分析回归目标值的差异
print '' 
print 'the max target value is ',numpy.max(y)
print 'the min target value is ',numpy.min(y)
print 'the average target value is ',numpy.average(y)

#标准化
from sklearn.preprocessing import StandardScaler
ss_x = StandardScaler()
ss_y = StandardScaler()
x_train = ss_x.fit_transform(x_train)
x_test = ss_x.transform(x_test)

y_train = ss_y.fit_transform(y_train)
y_test = ss_y.transform(y_test)

#使用回归树对波士顿房价进行预测
from sklearn.tree import DecisionTreeRegressor
dtr = DecisionTreeRegressor()
dtr.fit(x_train,y_train)
dtr_y_predict = dtr.predict(x_test)


#模型评价
print 'the value of default measurement of DecisionTreeRegressor is ',dtr.score(x_test,y_test)
from sklearn.metrics import r2_score,mean_squared_error,mean_absolute_error
print 'the value of R-squared of DecisionTreeRegressor is ',r2_score(y_test,dtr_y_predict)
print 'the value of mean squared error of DecisionTreeRegressor is ',mean_squared_error(ss_y.inverse_transform(y_test),ss_y.inverse_transform(dtr_y_predict))
print 'the value of mean absoluate error of DecisionTreeRegressor is ',mean_absolute_error(ss_y.inverse_transform(y_test),ss_y.inverse_transform(dtr_y_predict))
