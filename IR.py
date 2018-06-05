
# coding: utf-8

# In[1]:


import numpy as np
from __future__ import print_function
import math
import os
from io import open
import unicodedata
import string
import re
import random
import operator


# In[2]:


class doc_processor(object):
    
    def __init__(self, name):
        
        self.name = name
        self.word2idx = {}
        self.idx2word = {}
        self.word2count = {}
        self.n_words = 0
        
        self.doc = open('{}.txt'.format(name), encoding='utf-8').read().strip().split(' ')
        self.len = len(self.doc)
        self.addWord()
    
    def addWord(self):
        for word in self.doc:
            if word not in self.word2idx:
                self.word2idx[word] = self.n_words
                self.word2count[word] = 1
                self.idx2word[self.n_words] = word
                self.n_words += 1
            else:
                self.word2count[word] += 1
    
    def term_freq(self, term): # doc 에서 term이 몇번이나 나왔는지.
        if term not in self.word2idx:
            return 0
        return self.word2count[term]
        
    def tf_weight(self, term):
        if self.term_freq(term) > 0:
            return 1 + math.log10(self.term_freq(term))
        else:
            return 0
        
    def __len__(self):
        return self.len


# In[3]:


class query_processor(object):
    
    def __init__(self, query):
        
        self.word2idx = {}
        self.idx2word = {}
        self.word2count = {}
        self.n_words = 0
        
        self.doc = query
        self.len = len(self.doc)
        self.addWord()
    
    def addWord(self):
        for word in self.doc:
            if word not in self.word2idx:
                self.word2idx[word] = self.n_words
                self.word2count[word] = 1
                self.idx2word[self.n_words] = word
                self.n_words += 1
            else:
                self.word2count[word] += 1
    
    def term_freq(self, term): # doc 에서 term이 몇번이나 나왔는지.
        if term not in self.word2idx:
            return 0
        return self.word2count[term]
        
    def tf_weight(self, term):
        if self.term_freq(term) > 0:
            return 1 + math.log10(self.term_freq(term))
        else:
            return 0

    def __len__(self):
        return self.len


# In[4]:


def doc_freq(term, docs):
    ret = 0
    for doc in docs:
        if doc.term_freq(term) != 0:
            ret += 1
    return ret

def idf_weight(term, docs):
    return math.log10(1.0 * len(docs) / 1.0 * doc_freq(term, docs))

def tf_idf_weight(doc, term, docs):
    return doc.tf_weight(term) * idf_weight(term, docs)

    
def score_for_given_query(query, docs):
    score = {}
    for i in range(len(docs)):
        for term in query:
            if 'doc{}'.format(i+1) not in score:
                score['doc{}'.format(i+1)] = tf_idf_weight(docs[i], term, docs)
            else:
                score['doc{}'.format(i+1)] += tf_idf_weight(docs[i], term, docs)
    
    return score

def cosine_score(query, query_, docs):
    
    score = {}
    
    for i in range(len(docs)):
        for term in query:
            if 'doc{}'.format(i+1) not in score:
                score['doc{}'.format(i+1)] = tf_idf_weight(docs[i], term, docs) * (query_.tf_weight(term))
            else:
                score['doc{}'.format(i+1)] += tf_idf_weight(docs[i], term, docs) * (query_.tf_weight(term))
    
    for i in range(0,len(docs)):
        score['doc{}'.format(i+1)] /= 1.0 * docs[i].__len__()
        
    score = sorted(score.items(), key = operator.itemgetter(1), reverse = True)
    
    return score


# In[5]:


# using animal names for making docs and queries
animals = ["salmon", "tiger", "ant", "deer", "human", "fish", 'dog', 'cat', 'spider', 'rook', 'ram', 'raccoon', 'lark', 'hummingbird', 'gorilla', 'shark', 'whale', 'weasel', 'wolf', 'wren', 'yak', 'cow']


# In[6]:


# make 10 docs which consists of random 60 animals
for i in range(1,11):
    f = open('doc{}.txt'.format(i), 'w')
    random.seed(i*7)
    for k in range(0,60):
        f.write(unicode(animals[random.randrange(0,len(animals))] + ' '))
    f.close()

# pre-processing 10 docs
docs = []
for i in range(1,11):
    docs.append(doc_processor('doc{}'.format(i)))
    print('Pre-processed {}...'.format('doc{}'.format(i)))


# In[9]:


print('Start Testing...\n')

for i in range(5):
    print('\n############## Test{} ###############\n'.format(i+1))
    random.seed(i*3)
    
    #make random query(animal names)
    query = []
    
    for _ in range(5):
        query.append(animals[random.randrange(0,len(animals))])
    query_ = query_processor(query)
        
    print('Query : ', query,'\n')
    print('Compute tf.idf scores...\n')
    
    tfidf = score_for_given_query(query,docs)
    tfidf = sorted(tfidf.items(), key = operator.itemgetter(1), reverse = True)
    
    for k in range(1,6):
        print('Top{} : {}'.format(k, tfidf[k-1]))
        
    print('\nCompute cosine scores...\n')
    
    cosine_scr = cosine_score(query, query_, docs)
    
    for k in range(1,6):
        print('Top{} : {}'.format(k, cosine_scr[k-1]))
        
print('\n\nAll of tests are finished !!')

