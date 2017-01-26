from collections import defaultdict
from optparse import OptionParser

import numpy as np
from helper import Word2VecHelper
from sklearn import svm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import *
from sklearn.pipeline import Pipeline
import helper
from helper.VectorHelper import  *

classes = []


log = helper.enableLog()

def trainW2v(args):
    all = ["Accidents", "Arts", "Attacks", "Economy", "Miscellaneous", "Politics", "Science", "undefined"]

    if args.force == 1:
        files = ["./train/{}/{}/positive.txt".format(args.ontology, args.type),
                 "./train/{}/{}/negative.txt".format(args.ontology, args.type)]
        model = Word2VecHelper.createModel(files, name="{}_{}".format(args.ontology, args.type), merge=args.merge)
    else:
        model = Word2VecHelper.loadModel("{}_{}".format(args.ontology, args.type), merge=args.merge)

    w2v = {w: vec for w, vec in zip(model.wv.index2word, model.wv.syn0)}

    """
        model = Word2VecHelper.loadModel("{}_{}".format(args.ontology,args.type))
        w2v = {w: vec for w, vec in zip(model.wv.index2word, model.wv.syn0)}
    """


    train_instances, train_labels, train_texts = Word2VecHelper.loadData(all, args, 'train')
    test_instances, test_labels, test_texts = Word2VecHelper.loadData(all, args, 'test')

    C = 1.0  # SVM regularization parameter

    """
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
        """

    classifier_count = Pipeline(
    [
     ("tf_idf", TfidfEmbeddingVectorizer(w2v)),
     ('clf',svm.SVC(kernel='poly', C=C))]
    )

    parameters = {
                 'clf__gamma': (0.5, 0.7, 1,3),
                 'clf__degree': (1,5,10,15)}


    # x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.25)
    log.debug("BUilding the classifier {} with word count".format(args.classifier))
    #classifier_count.fit(x_train, y_train)
    gs_clf = GridSearchCV(classifier_count, parameters, n_jobs=-1)
    gs_clf = gs_clf.fit(train_texts, train_labels)
    y_pred = gs_clf.predict(test_texts)
    print(classification_report(test_labels, y_pred))
    for param_name in sorted(parameters.keys()):
        print("%s: %r" % (param_name, gs_clf.best_params_[param_name]))



if __name__ == "__main__":
    parser = OptionParser('''%prog -o ontology -t type -f force ''')
    parser.add_option('-o', '--ontology', dest='ontology', default="yago")
    parser.add_option('-t', '--type', dest='type', default="specific")
    parser.add_option('-f', '--force', dest='force', default=1, type=int)
    parser.add_option('-c', '--classifier', dest='classifier', default='nb')
    parser.add_option('-j', '--job', dest='job', type=int, default=10)
    parser.add_option('-w', '--window', dest='window', type=int, default=2)
    parser.add_option('-s', '--size', dest='size', type=int, default=300)
    parser.add_option('-m', '--merge', dest='merge', type=int, default=0)
    parser.add_option('-e', '--experiment', dest='experiment', type=int, default=0)

    opts, args = parser.parse_args()
    trainW2v(opts)