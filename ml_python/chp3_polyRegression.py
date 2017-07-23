#!/usr/bin/env python
# -*-encoding:utf-8-*-

x_train = [[6],[8],[10],[14],[18]]
y_train = [[7],[9],[13],[17.5],[18]]


#线性拟合
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train,y_train)

import numpy as np 
x_test = np.linspace(0,26,100)
x_test = x_test.reshape(x_test.shape[0],1)
y_predict = regressor.predict(x_test)

print 'the r-squared value of Linear Regressor performing on the training data is ',regressor.score(x_train,y_train)


#二次多项式拟合
from sklearn.preprocessing import PolynomialFeatures
poly2 = PolynomialFeatures(degree=2)
x_train_poly2 = poly2.fit_transform(x_train)
x_test_poly2 = poly2.transform(x_test)

regressor_poly2 = LinearRegression()
regressor_poly2.fit(x_train_poly2,y_train)
y_predict_poly2 = regressor_poly2.predict(x_test_poly2)

print 'the r-squared value of Ploynominal Regressor (Degree=2) performing on the training data is ',regressor_poly2.score(x_train_poly2,y_train)


#四次多项式
poly4 = PolynomialFeatures(degree=4)
x_train_poly4 = poly4.fit_transform(x_train)
x_test_poly4 = poly4.transform(x_test)

regressor_poly4 = LinearRegression()
regressor_poly4.fit(x_train_poly4,y_train)
y_predict_poly4 = regressor_poly4.predict(x_test_poly4)

print 'the r-squared value of Ploynominal Regressor (Degree=4) performing on the training data is ',regressor_poly4.score(x_train_poly4,y_train)


#展示
import matplotlib.pyplot as plt
plt.scatter(x_train,y_train)
plt1,=plt.plot(x_test,y_predict,label='Degree=1')
plt2,=plt.plot(x_test,y_predict_poly2,label='Degree=2')
plt4,=plt.plot(x_test,y_predict_poly4,label='Degree=4')
plt.axis([0,25,0,25])
plt.xlabel('Dismeter of pizza')
plt.ylabel('Price of pizza')
plt.legend(handles=[plt1,plt2,plt4])
plt.show()


#真实的测试数据集合
x_test = [[6],[8],[11],[16]]
y_test = [[8],[12],[15],[18]]

print ''
print 'the r-squared value of Linear Regressor performing on the test data is ',regressor.score(x_test,y_test)

x_test_poly2 = poly2.transform(x_test)
print 'the r-squared value of Ploynominal Regressor (Degree=2) performing on the test data is ',regressor_poly2.score(x_test_poly2,y_test)

x_test_poly4 = poly4.transform(x_test)
print 'the r-squared value of Ploynominal Regressor (Degree=4) performing on the test data is ',regressor_poly4.score(x_test_poly4,y_test)

