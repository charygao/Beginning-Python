#!/usr/bin/env python
# -*-encoding:utf-8-*-

sent1 = 'the cat is walking in the bedroom.'
sent2 = 'a dog was running across the kitchen.'

#词袋法（bag-of-words）对文本进行特征向量化
from sklearn.feature_extraction.text import CountVectorizer
count_vec = CountVectorizer()
sentences = [sent1,sent2]
print count_vec.fit_transform(sentences).toarray()
print count_vec.get_feature_names()
print ''

#使用NLTK对文不进行语言学分析
import nltk
tokens_1 = nltk.word_tokenize(sent1)
tokens_2 = nltk.word_tokenize(sent2)
print tokens_1
print tokens_2
print ''

#按照ascii排序输出
vocab_1 = sorted(set(tokens_1))
vocab_2 = sorted(set(tokens_2))
print vocab_1
print vocab_2
print ''

#寻找各个词汇最原始的词根
stemmer = nltk.stem.PorterStemmer()
stem_1 = [stemmer.stem(t) for t in tokens_1]
stem_2 = [stemmer.stem(t) for t in tokens_2]
print stem_1
print stem_2
print ''


#对每一个词汇进行词性标注
pos_tag_1 = nltk.tag.pos_tag(tokens_1)
pos_tag_2 = nltk.tag.pos_tag(tokens_2)
print pos_tag_1
print pos_tag_2
print ''