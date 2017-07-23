#!/usr/bin/env python
# -*-encoding:utf-8-*-

#本程序对比标准决策树、随机森林和梯度上升决策树三种模型

#获取数据
import pandas as pd
titanic = pd.read_csv('http://biostat.mc.vanderbilt.edu/wiki/pub/Main/DataSets/titanic.txt')
print titanic.head()
print titanic.info()


#机器学习中数据特征的选取，根据我们对Titanic事故的了解，sex,age,pclass这些特征都可能成为是否幸存的关键因素
x=titanic[['pclass','age','sex']]
y=titanic['survived']
print x.info()

#根据上面的输出：我们设计如下几个数据处理的任务
#（1）age,这个数据列只有633个，需要补完
#（2）sex和pclass两个数据列都是类别性的，需要转化位数值特征，用0/1代替

#（1）补充age数据，对于确实的年龄，采用平均值代替，这样在保证正常训练模型的同时，尽可能不影响预测任务
x['age'].fillna(x['age'].mean(),inplace=True)
print x.info()

#数据分割
from sklearn.cross_validation import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.25,random_state=33)

#特装转化
from sklearn.feature_extraction import DictVectorizer
vec = DictVectorizer(sparse=False)
x_train = vec.fit_transform(x_train.to_dict(orient='record'))
print vec.feature_names_  #查看转化后的数据特征列

x_test = vec.transform(x_test.to_dict(orient='record'))

#使用单一决策树分类器
from sklearn.tree import DecisionTreeClassifier
dtc = DecisionTreeClassifier()
dtc.fit(x_train,y_train)
dtc_y_predict  = dtc.predict(x_test)

#使用随机森林分类器进行集成模型的训练和预测
from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier()
rfc.fit(x_train,y_train)
rfc_y_predict = rfc.predict(x_test)


#使用梯度提升决策树进行集成模型的驯良和预测
from sklearn.ensemble import GradientBoostingClassifier
gbc = GradientBoostingClassifier()
gbc.fit(x_train,y_train)
gbc_y_predict = gbc.predict(x_test)

#使用xgboost模型
from xgboost import XGBClassifier
xgbc = XGBClassifier()
xgbc.fit(x_train,y_train)
xgbc_y_predict = xgbc.predict(x_test)


#性能评估
from sklearn.metrics import classification_report
print ''
print 'the accuracy of Decision Tree Classifier is ',dtc.score(x_test,y_test)
print classification_report(dtc_y_predict,y_test,target_names=['died','survived'])
print ''
print 'the accuracy of Random Forest Classifier is ',rfc.score(x_test,y_test)
print classification_report(rfc_y_predict,y_test,target_names=['died','survived'])
print ''
print 'the accuracy of Gradient boosting Classifier is ',gbc.score(x_test,y_test)
print classification_report(gbc_y_predict,y_test,target_names=['died','survived'])
print ''
print 'the accuracy of XGBClassifier is ',xgbc.score(x_test,y_test)
print classification_report(xgbc_y_predict,y_test,target_names=['died','survived'])


