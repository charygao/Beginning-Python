#!/usr/bin/env python
# -*-encoding:utf-8-*-

#读取手写数字图片数据集
from sklearn.datasets import load_digits
digits = load_digits()
print digits.data.shape

#数据集合的分割
from sklearn.cross_validation import train_test_split
x_train,x_test,y_train,y_test = train_test_split(digits.data,digits.target,test_size=0.25,random_state=33)
print "训练集合大小：",y_train.shape
print "测试集合大小：",y_test.shape

#使用支持向量分类器对数字图片进行识别
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

#特征数据进行标准化
ss= StandardScaler()
x_train = ss.fit_transform(x_train)
x_test = ss.fit_transform(x_test)

lsvc = LinearSVC()
lsvc.fit(x_train,y_train)
y_predict = lsvc.predict(x_test)


#模型评估
print "the accuracy of linear svc is ",lsvc.score(x_test,y_test)

from sklearn.metrics import classification_report
print classification_report(y_test,y_predict,target_names=digits.target_names.astype(str))



