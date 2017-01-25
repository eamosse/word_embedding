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
from sklearn.metrics import classification_report, confusion_matrix
from optparse import OptionParser
from nltk.tokenize import TweetTokenizer
import logging
from nltk.corpus import stopwords
import string
import sys
import nltk
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from nltk import ngrams
from sklearn.model_selection import GridSearchCV
import os
from helper import Word2VecHelper


classes = []


log = enableLog()

def trainW2v(args):
    if args.force == 1 :
        log.debug("Building the W2V model")
        files = ["./train/{}/{}/positive.txt".format(args.ontology,args.type),
                 "./train/{}/{}/negative.txt".format(args.ontology,args.type)]
        model = Word2VecHelper.createModel(name="{}_{}".format(args.ontology, args.type), files, args=args)
        w2v = {w: vec for w, vec in zip(model.wv.index2word, model.wv.syn0)}
        model.save("{}_{}.w2v".format(args.ontology,args.type))
    else:
        w2v = Word2VecHelper.loadModel("{}_{}".format(args.ontology,args.type))
    """with open(GLOVE_6B_200D_PATH, "r") as lines:
        word2vec = {line.split()[0]: np.array([float(i) for i in line.split()[1:]])
                    for line in lines}

    w2v = {**w2v, **word2vec, }
    """
    #x_train, y_train = loadData("train/{}/{}/positive.txt".format(args.ontology,args.type), "train/{}/{}/negative.txt".format(args.ontology,args.type))

    #x_test, y_test = loadData("test/{}/{}/positive.txt".format(args.ontology,args.type), "test/{}/{}/negative.txt".format(args.ontology,args.type))

    #x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

    x_train, y_train, x_test, y_test = loadData(args)
    C = 2.0  # SVM regularization parameter

    if args.classifier == 'poly':
        classifier_count = Pipeline([("word2vec vectorizer", MeanEmbeddingVectorizer(w2v)),
                          ("extra trees", svm.SVC(kernel="poly", degree=3, C=C))])

        classifier_tfidf = Pipeline([("word2vec vectorizer", TfidfEmbeddingVectorizer(w2v)),
                          ("extra trees", svm.SVC(kernel="poly", degree=3,C=C))])
    elif args.classifier == 'linear':
        classifier_count = Pipeline([("word2vec vectorizer", MeanEmbeddingVectorizer(w2v)),
                                     ("extra trees", svm.SVC(kernel="linear", C=C))])

        classifier_tfidf = Pipeline([("word2vec vectorizer", TfidfEmbeddingVectorizer(w2v)),
                                     ("extra trees", svm.SVC(kernel="linear", C=C))])
    elif args.classifier == 'rbf':
        classifier_count = Pipeline([("word2vec vectorizer", MeanEmbeddingVectorizer(w2v)),
                                     ("extra trees", svm.SVC(kernel='rbf', gamma=0.7, C=C))])

        classifier_tfidf = Pipeline([("word2vec vectorizer", TfidfEmbeddingVectorizer(w2v)),
                                     ("extra trees", svm.SVC(kernel='rbf', gamma=0.7, C=C))])
    elif args.classifier == 'ben':
        classifier_count = Pipeline([("word2vec vectorizer", MeanEmbeddingVectorizer(w2v)),
                                    ("extra trees", BernoulliNB())])

        classifier_tfidf = Pipeline([("word2vec vectorizer", TfidfEmbeddingVectorizer(w2v)),
                                ("extra trees", BernoulliNB())])
    else:
        classifier_count = Pipeline(
        [
         ("tf_idf", TfidfEmbeddingVectorizer(w2v)),
         ('clf',svm.SVC())]
        )

        classifier_tfidf = Pipeline(
        [("tfidf_vectorizer", TfidfVectorizer(analyzer=lambda x: x)), ("multinomial nb", MultinomialNB())])

    parameters = {
                 'clf__kernel': ('linear', 'rbf', 'poly'),
                 'clf__gamma': (0.1, 0.3, 0.5, 0.7, 0.9),
                 'clf__degree': (1,10,100),
                 'clf__C': (1.0, 3.0, 5.0)}


    # x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.25)
    log.debug("BUilding the classifier {} with word count".format(args.classifier))
    #classifier_count.fit(x_train, y_train)
    gs_clf = GridSearchCV(classifier_count, parameters, n_jobs=-1)
    gs_clf = gs_clf.fit(x_train, y_train)
    y_pred = gs_clf.predict(x_test)
    print(classification_report(y_test, y_pred))
    for param_name in sorted(parameters.keys()):
        print("%s: %r" % (param_name, gs_clf.best_params_[param_name]))

    """
    y_pred = classifier_count.predict(x_test)
    print(classification_report(y_test, y_pred))
    print(confusion_matrix(y_test,y_pred, labels=classes))

    log.debug("BUilding the classifier {} with tfidf".format(args.classifier))
    classifier_tfidf.fit(x_train, y_train)
    y_pred = classifier_tfidf.predict(x_test)
    print(classification_report(y_test, y_pred))
    print(confusion_matrix(y_test, y_pred, labels=classes))
    """

    #print((cross_val_score(etree_w2v, X, y, cv=10).mean()))


class MeanEmbeddingVectorizer(object):
    def __init__(self, value):
        self.word2vec = value
        #print(word2vec.values())
        self.dim = len(next(iter(value.values())))

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

    def get_params(self, deep=True):
        # suppose this estimator has parameters "alpha" and "recursive"
        return {"word2vec": self.word2vec}

    """def set_params(self, **parameters):
        for parameter, value in parameters.items():
            self.setattr(parameter, value)
        return self`
    """

    def transform(self, X):
        return np.array([
                            np.mean([self.word2vec[w] * self.word2weight[w]
                                     for w in words if w in self.word2vec] or
                                    [np.zeros(self.dim)], axis=0)
                            for words in X
                            ])

#trainW2v()

if __name__ == "__main__":
    parser = OptionParser('''%prog -o ontology -t type -f force ''')
    parser.add_option('-o', '--ontology', dest='ontology', default="dbpedia")
    parser.add_option('-t', '--type', dest='type', default="generic")
    parser.add_option('-f', '--force', dest='force', default=1, type=int)
    parser.add_option('-c', '--classifier', dest='classifier', default='nb')
    parser.add_option('-j', '--job', dest='job', type=int, default=10)
    parser.add_option('-w', '--window', dest='window', type=int, default=2)
    parser.add_option('-s', '--size', dest='size', type=int, default=300)
    parser.add_option('-m', '--min', dest='min_count', type=int, default=5)
    parser.add_option('-e', '--experiment', dest='experiment', type=int, default=0)
    opts, args = parser.parse_args()
    trainW2v(opts)