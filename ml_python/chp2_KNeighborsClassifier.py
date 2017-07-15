#!/usr/bin/env python
# -*-encoding:utf-8-*-

#数据集合
from sklearn.datasets import load_iris
iris = load_iris()
##数据规模
print iris.data.shape
##数据说明
print iris.DESCR

#数据集合分割
from sklearn.cross_validation import train_test_split
x_train,x_test,y_train,y_test = train_test_split(iris.data,iris.target,test_size=0.25,random_state=33)

#使用K近邻分类器对iris数据进行类别预测
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier

ss= StandardScaler()
x_train = ss.fit_transform(x_train)
x_test = ss.transform(x_test)

knc = KNeighborsClassifier()
knc.fit(x_train,y_train)
y_predict = knc.predict(x_test)

#性能评估
from sklearn.metrics import classification_report
print "the accuracy of k_nearest neighbor classifier is ",knc.score(x_test,y_test)
print classification_report(y_test,y_predict,target_names=iris.target_names)
