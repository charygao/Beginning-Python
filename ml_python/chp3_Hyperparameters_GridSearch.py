#!/usr/bin/env python
# -*-encoding:utf-8-*-

from sklearn.datasets import fetch_20newsgroups
import numpy as np 
news  =fetch_20newsgroups(subset='all')

from sklearn.cross_validation import train_test_split
x_train,x_test,y_train,y_test = train_test_split(news.data[:3000],news.target[:3000],test_size=0.25,random_state=33)

from sklearn.svm import SVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
clf = Pipeline([('vect',TfidfVectorizer(stop_words='english',analyzer='word')),('svc',SVC())])
parameters = {'svc__gamma':np.logspace(-2,1,4),'svc__C':np.logspace(-1,1,3)}

from sklearn.grid_search import GridSearchCV
gs = GridSearchCV(clf,parameters,verbose=2,refit=True,cv=3,n_jobs=4)

gs.fit(x_train,y_train)
print '最佳模型参数：',gs.best_params_,gs.best_score_
print '测试数据集合上的模型性能：',gs.score(x_test,y_test)
