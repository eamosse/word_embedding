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
import FileHelper
from nltk.tokenize import TweetTokenizer
import logging
from nltk.corpus import stopwords
import string
import sys
import nltk
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from nltk import ngrams
classes = []

log = logging.getLogger()
log.setLevel(logging.DEBUG)

ch = logging.StreamHandler(sys.stdout)
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)
log.addHandler(ch)

tknzr = TweetTokenizer()
stop = stopwords.words('english') + list(string.punctuation)
porter = nltk.PorterStemmer()
GLOVE_6B_200D_PATH = "glove.twitter.27B/glove.twitter.27B.200d.txt"
"""
TODO:Try word context by taking n tokens before and after each token in the sentence
REF:https://www.ijcai.org/Proceedings/16/Papers/401.pdf

"""

def tokenize(text):
    tokens = [token for token in tknzr.tokenize(text.lower()) if token not in stop and len(token) > 2]
    #_grams = ngrams(tokens, 3)
    #tokens = []
    #for g in _grams:
        #tokens.append(' '.join(g))
    return tokens


class MySentences(object):
    def __init__(self, dirname):
        self.dirname = dirname

    def __iter__(self):
        for fname in os.listdir(self.dirname):
            for line in open(os.path.join(self.dirname, fname)):
                yield tokenize(line)


def createWord2VecModel(directory="train", args={}):
    sentences = MySentences(directory)  # a memory-friendly iterator
    model = gensim.models.Word2Vec(sentences, workers=args.job, size=args.size, min_count=args.min_count, window=args.window)

    return model


def mFile(file):
    X = []
    with open(file, "r") as infile:
        for line in infile:
            # label, text = line.split("\t")
            # texts are already tokenized, just split on space
            # in a real case we would use e.g. spaCy for tokenization
            # and maybe remove stopwords etc.
            X.append(tokenize(line))
    return X


def load(folder, X, y,label):
    train = mFile(folder)

    if len(train) > 0 :
        y.extend([label for _ in train])
        X.extend(train)
    return X, y

def loadData(args):
    X_train, y_train, X_test, y_test = [], [], [], []
    global  classes
    if args.experiment == 1:
        classes = ["Science", "Attacks", "Politics", "Arts", "Sports", "Accidents", "Economy"]
    else:
        classes = ['negative', 'positive']
    for index, category in enumerate(classes):
        print("loading", category)
        load("train/{}/{}/{}.txt".format(args.ontology,args.type,category),X_train,y_train,category)
        load("test/{}/{}/{}.txt".format(args.ontology, args.type, category), X_test, y_test, category)
        #y_train.extend([index for _ in train])
        #X_train.extend(train)

    X_train, y_train, X_test, y_test = np.array(X_train), np.array(y_train),np.array(X_test), np.array(y_test)
    return X_train, y_train, X_test, y_test

def loadModel(name):
    model = gensim.models.Word2Vec.load(name)
    return {w: vec for w, vec in zip(model.wv.index2word, model.wv.syn0)}

def trainW2v(args):
    if args.force == 1 :
        log.debug("Generatiing files....")
        FileHelper.generate(args.type,args.ontology)
        log.debug("Building the W2V model")
        model = createWord2VecModel("train", args=args)
        w2v = {w: vec for w, vec in zip(model.wv.index2word, model.wv.syn0)}
        model.save("{}_{}.w2v".format(args.ontology,args.type))
    else:
        w2v = loadModel("{}_{}.w2v".format(args.ontology,args.type))
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
        [("count_vectorizer", CountVectorizer(analyzer=lambda x: x)), ("multinomial nb", MultinomialNB())])
        classifier_tfidf = Pipeline(
        [("tfidf_vectorizer", TfidfVectorizer(analyzer=lambda x: x)), ("multinomial nb", MultinomialNB())])

    #x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.25)
    log.debug("BUilding the classifier {} with word count".format(args.classifier))
    classifier_count.fit(x_train, y_train)
    y_pred = classifier_count.predict(x_test)
    print(classification_report(y_test, y_pred))
    print(confusion_matrix(y_test,y_pred, labels=classes))

    log.debug("BUilding the classifier {} with tfidf".format(args.classifier))
    classifier_tfidf.fit(x_train, y_train)
    y_pred = classifier_tfidf.predict(x_test)
    print(classification_report(y_test, y_pred))
    print(confusion_matrix(y_test, y_pred, labels=classes))

    #print((cross_val_score(etree_w2v, X, y, cv=10).mean()))


class MeanEmbeddingVectorizer(object):
    def __init__(self, word2vec):
        self.word2vec = word2vec
        #print(word2vec.values())
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

#trainW2v()

if __name__ == "__main__":
    parser = OptionParser('''%prog -o ontology -t type -f force ''')
    parser.add_option('-o', '--ontology', dest='ontology', default="yago")
    parser.add_option('-t', '--type', dest='type', default="specific")
    parser.add_option('-f', '--force', dest='force', default=0, type=int)
    parser.add_option('-c', '--classifier', dest='classifier', default='linear')
    parser.add_option('-j', '--job', dest='job', type=int, default=10)
    parser.add_option('-w', '--window', dest='window', type=int, default=2)
    parser.add_option('-s', '--size', dest='size', type=int, default=300)
    parser.add_option('-m', '--min', dest='min_count', type=int, default=5)
    parser.add_option('-e', '--experiment', dest='experiment', type=int, default=1)
    opts, args = parser.parse_args()
    trainW2v(opts)