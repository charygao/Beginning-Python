#!/usr/bin/env python
# -*-encoding:utf-8-*-

#DictVectorizer对特征进行抽取和向量化
measurements = [{'City':'Dubai','Temperature':33.},{'City':'London','Temperature':12.},{'City':'San Fransisco','Temperature':18.}]

from sklearn.feature_extraction import DictVectorizer
vec = DictVectorizer()
featureVect = vec.fit_transform(measurements).toarray()
print vec.get_feature_names()
print featureVect
print ''


#CountVectorizer
#使用CountVectorizer并且不去掉停用词汇（Stop Words）的情况下，对文本特征进行量化的朴素贝叶斯分类性能测试
from sklearn.datasets import fetch_20newsgroups
news = fetch_20newsgroups(subset='all')

from sklearn.cross_validation import train_test_split
x_train,x_test,y_train,y_test = train_test_split(news.data,news.target,test_size=0.25,random_state=33)

from sklearn.feature_extraction.text import CountVectorizer
count_vec = CountVectorizer()
x_count_train = count_vec.fit_transform(x_train)
x_count_test = count_vec.transform(x_test)

from sklearn.naive_bayes import MultinomialNB
mnb_count = MultinomialNB()
mnb_count.fit(x_count_train,y_train)
y_count_predict = mnb_count.predict(x_count_test)


from sklearn.metrics import classification_report
print " the accuracy of classifying 20newsgroups using naive bayes (CountVectorizer without filtering stopwords): ",mnb_count.score(x_count_test,y_test)
print classification_report(y_test,y_count_predict,target_names=news.target_names)
print ''

#TfidfVectorizer
#使用TfidfVectorizer并且不去掉停用词汇（Stop Words）的情况下，对文本特征进行量化的朴素贝叶斯分类性能测试
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf_vec = TfidfVectorizer()
x_tfidf_train = tfidf_vec.fit_transform(x_train)
x_tfidf_test = tfidf_vec.transform(x_test)

mnb_tfidf = MultinomialNB()
mnb_tfidf.fit(x_tfidf_train,y_train)
y_tfidf_predict = mnb_tfidf.predict(x_tfidf_test)

print " the accuracy of classifying 20newsgroups using naive bayes (TfidfVectorizer without filtering stopwords): ",mnb_tfidf.score(x_tfidf_test,y_test)
print classification_report(y_test,y_tfidf_predict,target_names=news.target_names)
print ''


#在过滤掉停用词（Stop Words）的情况下CountVectorizer
count_filter_vec = CountVectorizer(analyzer='word',stop_words='english')
x_count_filter_train = count_filter_vec.fit_transform(x_train)
x_count_filter_test = count_filter_vec.transform(x_test)

mnb_count_filter = MultinomialNB()
mnb_count_filter.fit(x_count_filter_train,y_train)
y_count_filter_predict = mnb_count_filter.predict(x_count_filter_test)

from sklearn.metrics import classification_report
print " the accuracy of classifying 20newsgroups using naive bayes (CountVectorizer with filtering stopwords): ",mnb_count_filter.score(x_count_filter_test,y_test)
print classification_report(y_test,y_count_filter_predict,target_names=news.target_names)
print ''


#在过滤掉停用词（Stop Words）的情况下TfidfVectorizer
tfidf_filter_vec = TfidfVectorizer(analyzer='word',stop_words='english')
x_tfidf_filter_train = tfidf_filter_vec.fit_transform(x_train)
x_tfidf_filter_test = tfidf_filter_vec.transform(x_test)

mnb_tfidf_filter = MultinomialNB()
mnb_tfidf_filter.fit(x_tfidf_filter_train,y_train)
y_tfidf_filter_predict = mnb_tfidf_filter.predict(x_tfidf_filter_test)

print " the accuracy of classifying 20newsgroups using naive bayes (TfidfVectorizer with filtering stopwords): ",mnb_tfidf_filter.score(x_tfidf_filter_test,y_test)
print classification_report(y_test,y_tfidf_filter_predict,target_names=news.target_names)
print ''
