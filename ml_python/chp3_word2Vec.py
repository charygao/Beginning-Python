#!/usr/bin/env python
# -*-encoding:utf-8-*-

from sklearn.datasets import fetch_20newsgroups
allnews = fetch_20newsgroups(subset='all')
x,y=allnews.data,allnews.target

from bs4 import BeautifulSoup
import nltk,re


def news_to_sentences(news):
    news_text = BeautifulSoup(news).get_text()
    tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
    raw_sentences = tokenizer.tokenize(news_text)
    sentences = []
    for sent in raw_sentences:
        sentences.append(re.sub('[^a-zA-Z]',' ',sent.lower().strip()).split())
    return sentences


sentences = []
for news in x:
    sentences+=news_to_sentences(news)


from gensim.models import word2vec
num_features = 300 #配置词向量的维度
min_word_counbt = 20 # 保证被考虑的词汇的频度
num_workers = 2 #并行时使用的CPU的核心数量
context = 5 #定义训练词向量的上下文的窗口的大小
downsampling = 1e-3

model = word2vec.Word2Vec(sentences,workers=num_workers,size=num_features,min_count=min_word_counbt,window=context,sample=downsampling)
model.init_sims(replace=True)

print model.most_similar('morning')
print ''
print model.most_similar('email')
print ''