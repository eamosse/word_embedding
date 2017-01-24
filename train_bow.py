import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import data_helpers as data_helpers
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import  train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import *
from sklearn.feature_extraction.text import *
import logging
import sys
from nltk.tokenize import TweetTokenizer
from nltk.corpus import stopwords
import string
import sys
from sklearn.naive_bayes import *
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
import os
import time

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import svm
from sklearn.metrics import classification_report

stop = stopwords.words('english') + list(string.punctuation)

tknzr = TweetTokenizer()
def tokenize(text):
    tokens = [token for token in tknzr.tokenize(text.lower()) if token not in stop and len(token) > 2]
    #_grams = ngrams(tokens, 3)
    #tokens = []
    #for g in _grams:
        #tokens.append(' '.join(g))
    return tokens

def mFile(file):
    X = []
    with open(file, "r") as infile:
        for line in infile:
            X.append(' '.join(tokenize(line)))
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

log = logging.getLogger()
log.setLevel(logging.DEBUG)

ch = logging.StreamHandler(sys.stdout)
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)
log.addHandler(ch)

log.debug("Loading data")
train_data, train_labels = loadData("train/dbpedia/generic/positive.txt", "train/dbpedia/generic/negative.txt")
test_data, test_labels = loadData("test/dbpedia/generic/positive.txt", "test/dbpedia/generic/negative.txt")



parameters = {'vect__ngram_range': [(1, 1), (1, 2)],
               'tfidf__use_idf': (True, False),
               'clf__alpha': (1e-2, 1e-3),
 }

text_clf = Pipeline([('vect', CountVectorizer()),
                     ('tfidf', TfidfTransformer()),
                      ('clf', BernoulliNB()),
 ])

gs_clf = GridSearchCV(text_clf, parameters, n_jobs=-1)

gs_clf = gs_clf.fit(train_data,train_labels)
y_pred = gs_clf.predict(test_data)
print(classification_report(test_labels, y_pred))

for param_name in sorted(parameters.keys()):
    print("%s: %r" % (param_name, gs_clf.best_params_[param_name]))


"""
# x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.25)
log.debug("BUilding the classifier {} with word count".format(args.classifier))
# classifier_count.fit(x_train, y_train)
gs_clf = GridSearchCV(classifier_count, parameters, n_jobs=-1)
gs_clf = gs_clf.fit(x_train, y_train)
for param_name in sorted(parameters.keys()):
    print("%s: %r" % (param_name, gs_clf.best_params_[param_name]))


# Create feature vectors
vectorizer = TfidfVectorizer(min_df=5,
                             max_df = 0.8,
                             sublinear_tf=True,
                             use_idf=True)
train_vectors = vectorizer.fit_transform(train_data)
test_vectors = vectorizer.transform(test_data)

# Perform classification with SVM, kernel=rbf
classifier_rbf = svm.SVC(gamma=0.7, C=1.0)
t0 = time.time()
classifier_rbf.fit(train_vectors, train_labels)
t1 = time.time()
prediction_rbf = classifier_rbf.predict(test_vectors)
t2 = time.time()
time_rbf_train = t1-t0
time_rbf_predict = t2-t1

# Perform classification with SVM, kernel=linear
classifier_linear = svm.SVC(kernel='poly', gamma=0.5)
t0 = time.time()
classifier_linear.fit(train_vectors, train_labels)
t1 = time.time()
prediction_linear = classifier_linear.predict(test_vectors)
t2 = time.time()
time_linear_train = t1-t0
time_linear_predict = t2-t1

# Perform classification with SVM, kernel=linear
classifier_liblinear = svm.LinearSVC()
t0 = time.time()
classifier_liblinear.fit(train_vectors, train_labels)
t1 = time.time()
prediction_liblinear = classifier_liblinear.predict(test_vectors)
t2 = time.time()
time_liblinear_train = t1-t0
time_liblinear_predict = t2-t1

# Print results in a nice table
print("Results for SVC(kernel=rbf)")
print("Training time: %fs; Prediction time: %fs" % (time_rbf_train, time_rbf_predict))
print(classification_report(test_labels, prediction_rbf))
print("Results for SVC(kernel=linear)")
print("Training time: %fs; Prediction time: %fs" % (time_linear_train, time_linear_predict))
print(classification_report(test_labels, prediction_linear))
print("Results for LinearSVC()")
print("Training time: %fs; Prediction time: %fs" % (time_liblinear_train, time_liblinear_predict))
print(classification_report(test_labels, prediction_liblinear))
"""