#!/usr/bin/env python
# -*-encoding:utf-8-*-

import tensorflow as tf 
import numpy as np 

#Demo: Hello World
greeting = tf.constant('Hello Google Tensorflow!')
sess = tf.Session()
result = sess.run(greeting)
print result
sess.close()


#Demo:使用Tensorflow完成一次线性函数计算
matrix1 = tf.constant([[3.,3.]])
matrix2 = tf.constant([[2.],[2.]])
product = tf.matmul(matrix1,matrix2)
linear = tf.add(product,tf.constant(2.0))
with tf.Session() as sess:
    result = sess.run(linear)
    print result


#Demo:使用Tensorflow自定义一个线性分类器，对“良/恶性乳腺癌肿瘤”进行预测
import tensorflow as tf 
import numpy as np 
import pandas as pd 

trainData = pd.read_csv('./DataSet/breast-cancer-train.csv')
testData = pd.read_csv('./DataSet/breast-cancer-test.csv')

x_train = np.float32(trainData[['Clump Thickness','Cell Size']].T)
y_train = np.float32(trainData['Type'].T)
x_test = np.float32(testData[['Clump Thickness','Cell Size']].T)
y_test = np.float32(testData['Type'].T)

#定义线性模型的截距b
b=tf.Variable(tf.zeros([1.0]))
#定义线性模型的系数W，并设置初始值是-1~1之间的随机值
W= tf.Variable(tf.random_uniform([1,2],-1.0,1.0))
#显式定义线性函数
y = tf.matmul(W,x_train)+b

#使用reduce_mean取得训练集上的均方误差
loss = tf.reduce_mean(tf.square(y-y_train))

#使用梯度下降估计参数W，b，并设置迭代步长为0.01，和Scikit-learn中的SGDRegressor类似
optimizer = tf.train.GradientDescentOptimizer(0.01)

#以最小二乘损失为优化目标
train = optimizer.minimize(loss)

#初始化所有变量
init= tf.global_variables_initializer()

sess = tf.Session()
sess.run(init)

#迭代1000轮次，训练参数
for step in xrange(0,1000):
    sess.run(train)
    if(step % 200 == 0):
        print step,sess.run(W),sess.run(b)
    



#测试数据
test_negative = testData.loc[testData['Type']==0][['Clump Thickness','Cell Size']]
test_positive = testData.loc[testData['Type']==1][['Clump Thickness','Cell Size']]

import matplotlib.pyplot as plt 
plt.scatter(test_negative['Clump Thickness'],test_negative['Cell Size'],marker='o',s=200,c='red')
plt.scatter(test_positive['Clump Thickness'],test_positive['Cell Size'],marker='x',s=150,c='black')
plt.xlabel('Clump Thinkness')
plt.ylabel('Cell Size')

lx = np.arange(0,12)
ly = (0.5-sess.run(b)-lx*sess.run(W)[0][0])/sess.run(W)[0][1]
plt.plot(lx,ly,color = 'green')
plt.show()

