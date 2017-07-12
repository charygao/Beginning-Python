#!/usr/bin/env python
# -*-encoding:utf-8-*-


#数据处理
import pandas as pd
import numpy as np

column_names = ['Sample code number','Clump Thickness','Uniformity of Cell Size','Uniformity of Cell Shape','Marginal Adhesion','Single Epithelial Cell Size','Bare Nuclei','Bland Chromatin','Normal Nucleoli','Mitoses','Class']
data=pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/breast-cancer-wisconsin.data',names=column_names)

data = data.replace(to_replace='?',value=np.nan)
data = data.dropna(how='any')
print data.shape

#拆分数据集为训练集和测试集
from sklearn.cross_validation import train_test_split
x_train,x_test,y_train,y_test=train_test_split(data[column_names[1:10]],data[column_names[10]],test_size=0.25,random_state=33)
print y_train.value_counts()
print y_test.value_counts()


#模型训练和预测
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier

##标准化数据
ss = StandardScaler()
x_train = ss.fit_transform(x_train)
x_test = ss.fit_transform(x_test)

##logisticRegression
lr = LogisticRegression()
lr.fit(x_train,y_train)
lr_y_predict = lr.predict(x_test)

##SGDClassifier
sgdc= SGDClassifier()
sgdc.fit(x_train,y_train)
sgdc_y_predict = sgdc.predict(x_test)

#模型的性能评估
from sklearn.metrics import classification_report
print "Accuracy of lr classifier: ",lr.score(x_test,y_test)
print classification_report(y_test,lr_y_predict,target_names=['Benign','Malignant'])

print "Accuracy of sgdc classifier: ",sgdc.score(x_test,y_test)
print classification_report(y_test,sgdc_y_predict,target_names=['Benign','Malignant'])

