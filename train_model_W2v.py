import os

import gensim
import numpy as np
from sklearn.cross_validation import *
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import *
from sklearn.linear_model import *
from sklearn import svm
from sklearn.pipeline import Pipeline
from collections import Counter, defaultdict
from sklearn.metrics import classification_report

class MySentences(object):
    def __init__(self, dirname):
        self.dirname = dirname

    def __iter__(self):
        for fname in os.listdir(self.dirname):
            for line in open(os.path.join(self.dirname, fname)):
                yield line.split()


def createWord2VecModel(directory="train"):
    sentences = MySentences(directory)  # a memory-friendly iterator
    model = gensim.models.Word2Vec(sentences)
    #model.save("./{}.w2v".format(directory))
    return model


def mFile(file):
    X = []
    with open(file, "r") as infile:
        for line in infile:
            # label, text = line.split("\t")
            # texts are already tokenized, just split on space
            # in a real case we would use e.g. spaCy for tokenization
            # and maybe remove stopwords etc.
            X.append(line.split())
    return X


def loadData(positive,negative):
    X, y = [], []
    p = mFile(positive)
    y.extend([1 for _ in p])
    X.extend(p)

    n = mFile(negative)
    y.extend([0 for _ in n])
    X.extend(n)
    X, y = np.array(X), np.array(y)
    return X, y

def trainW2v():
    model = createWord2VecModel("train")
    w2v = {w: vec for w, vec in zip(model.index2word, model.syn0)}
    x_train, y_train = loadData("train/positive.txt", "train/negative.txt")
    x_test, y_test = loadData("test/positive.txt", "test/negative.txt")
    svm_count = Pipeline([("word2vec vectorizer", MeanEmbeddingVectorizer(w2v)),
                          ("extra trees", svm.SVC(kernel="poly", degree=3))])

    svm_tfidf = Pipeline([("word2vec vectorizer", TfidfEmbeddingVectorizer(w2v)),
                          ("extra trees", svm.SVC(kernel="poly", degree=3))])
    """
    etree_w2v_tfidf = Pipeline([("word2vec vectorizer", TfidfEmbeddingVectorizer(w2v)),
                                ("extra trees", MultinomialNB())])
    etree_w2v_count = Pipeline([("word2vec vectorizer", MeanEmbeddingVectorizer(w2v)),
                                ("extra trees", MultinomialNB())])


    """
    #x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

    svm_count.fit(x_train, y_train)
    y_pred = svm_count.predict(x_test)
    print(classification_report(y_test, y_pred))

    svm_tfidf.fit(x_train, y_train)
    y_pred = svm_tfidf.predict(x_test)
    print(classification_report(y_test, y_pred))

    #print((cross_val_score(etree_w2v, X, y, cv=10).mean()))


class MeanEmbeddingVectorizer(object):
    def __init__(self, word2vec):
        self.word2vec = word2vec
        self.dim = len(next(iter(word2vec.values())))

    def fit(self, X, y):
        return self

    def transform(self, X):
        return np.array([
                            np.mean([self.word2vec[w] for w in words if w in self.word2vec]
                                    or [np.zeros(self.dim)], axis=0)
                            for words in X
                            ])


# and a tf-idf version of the same

class TfidfEmbeddingVectorizer(object):
    def __init__(self, word2vec):
        self.word2vec = word2vec
        self.word2weight = None
        self.dim = len(next(iter(word2vec.values())))

    def fit(self, X, y):
        tfidf = TfidfVectorizer(analyzer=lambda x: x)
        tfidf.fit(X)
        # if a word was never seen - it must be at least as infrequent
        # as any of the known words - so the default idf is the max of
        # known idf's
        max_idf = max(tfidf.idf_)
        self.word2weight = defaultdict(
            lambda: max_idf,
            [(w, tfidf.idf_[i]) for w, i in list(tfidf.vocabulary_.items())])

        return self

    def transform(self, X):
        return np.array([
                            np.mean([self.word2vec[w] * self.word2weight[w]
                                     for w in words if w in self.word2vec] or
                                    [np.zeros(self.dim)], axis=0)
                            for words in X
                            ])


trainW2v()
