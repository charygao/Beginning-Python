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

#使用支持向量机（回归）模型对波士顿房价进行预测
from sklearn.svm import SVR

#使用线性核函数配置
linear_svr = SVR(kernel='linear')
linear_svr.fit(x_test,y_test)
linear_svr_y_predict = linear_svr.predict(x_test)

#使用多项式核函数配置
poly_svr = SVR(kernel='poly')
poly_svr.fit(x_test,y_test)
poly_svr_y_predict = poly_svr.predict(x_test)

#使用径向基核函数配置
rbf_svr = SVR(kernel='rbf')
rbf_svr.fit(x_test,y_test)
rbf_svr_y_predict = rbf_svr.predict(x_test)



#模型评价
print 'the value of default measurement of Linear SVR is ',linear_svr.score(x_test,y_test)
from sklearn.metrics import r2_score,mean_squared_error,mean_absolute_error
print 'the value of R-squared of Linear SVR is ',r2_score(y_test,linear_svr_y_predict)
print 'the value of mean squared error of Linear SVR is ',mean_squared_error(ss_y.inverse_transform(y_test),ss_y.inverse_transform(linear_svr_y_predict))
print 'the value of mean absoluate error of Linear SVR is ',mean_absolute_error(ss_y.inverse_transform(y_test),ss_y.inverse_transform(linear_svr_y_predict))

print ''
print 'the value of default measurement of Poly SVR is ',poly_svr.score(x_test,y_test)
from sklearn.metrics import r2_score,mean_squared_error,mean_absolute_error
print 'the value of R-squared of Poly SVR is ',r2_score(y_test,poly_svr_y_predict)
print 'the value of mean squared error of Poly SVR is ',mean_squared_error(ss_y.inverse_transform(y_test),ss_y.inverse_transform(poly_svr_y_predict))
print 'the value of mean absoluate error of Poly SVR is ',mean_absolute_error(ss_y.inverse_transform(y_test),ss_y.inverse_transform(poly_svr_y_predict))

print ''
print 'the value of default measurement of RBF SVR is ',rbf_svr.score(x_test,y_test)
from sklearn.metrics import r2_score,mean_squared_error,mean_absolute_error
print 'the value of R-squared of RBF SVR is ',r2_score(y_test,rbf_svr_y_predict)
print 'the value of mean squared error of RBF SVR is ',mean_squared_error(ss_y.inverse_transform(y_test),ss_y.inverse_transform(rbf_svr_y_predict))
print 'the value of mean absoluate error of RBF SVR is ',mean_absolute_error(ss_y.inverse_transform(y_test),ss_y.inverse_transform(rbf_svr_y_predict))