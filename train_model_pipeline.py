from sklearn.naive_bayes import *
from sklearn import svm
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix
from optparse import OptionParser
from helper import FileHelper, Word2VecHelper, GraphHelper
import helper
import numpy as np
from sklearn.externals import joblib
from helper.VectorHelper import *
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import permutation_test_score

log = helper.enableLog()


def trainW2v(args):
    all = ["Accidents", "Arts", "Attacks", "Economy", "Miscellaneous", "Politics", "Science", "undefined"]
    binaries = ['negative', 'positive']

    if args.force == 1:
        files = ["./train/{}/{}/positive.txt".format(args.ontology, args.type),
                 "./train/{}/{}/negative.txt".format(args.ontology, args.type)]
        model = Word2VecHelper.createModel(files,name="{}_{}".format(args.ontology, args.type), merge=args.merge)
    else:
        model = Word2VecHelper.loadModel("{}_{}".format(args.ontology,args.type),merge=args.merge)

    w2v = {w: vec for w, vec in zip(model.wv.index2word, model.wv.syn0)}

    train_instances, train_labels, train_texts = Word2VecHelper.loadData(binaries,args, 'train')
    test_instances, test_labels, test_texts = Word2VecHelper.loadData(binaries,args, 'test')

    gamma = 0.5
    degree = 0.5
    cl = ["ben"]
    C = 2.0  # SVM regularization parameter

    for classifier in cl:
        args.classifier = classifier
        print("RUNNING FOR ", classifier)

        if args.classifier == 'poly':
            classifier = Pipeline([("word2vec vectorizer", MeanEmbeddingVectorizer(w2v)),
                                   ("extra trees",
                                    svm.SVC(kernel="poly", degree=degree, C=C, gamma=gamma, probability=True))])
        elif args.classifier == 'linear':
            classifier = Pipeline([("word2vec vectorizer", MeanEmbeddingVectorizer(w2v)),
                                   ("extra trees", svm.LinearSVC(C=C))])
        elif args.classifier == 'rbf':
            classifier = Pipeline([("word2vec vectorizer", MeanEmbeddingVectorizer(w2v)),
                                   ("extra trees", svm.SVC(kernel='rbf', gamma=gamma, C=C, probability=True))])
        elif args.classifier == 'ben':
            classifier = Pipeline([("word2vec vectorizer", MeanEmbeddingVectorizer(w2v)),
                                   ("extra trees", BernoulliNB())])
        else:
            classifier = Pipeline(
                [("count_vectorizer", CountVectorizer(analyzer=lambda x: x)), ("multinomial nb", MultinomialNB())])

        #x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.25)
        log.debug("Train the binary model".format(args.classifier))
        #classifier.fit(train_texts, train_labels)
        #y_pred = classifier.predict(test_texts)

        #print(classification_report(test_labels, y_pred))
        #print(confusion_matrix(test_labels,y_pred, labels=binaries))

        score, permutation_scores, pvalue = permutation_test_score(
        classifier, train_texts, train_labels, scoring="accuracy", cv=None, n_permutations=100, n_jobs=1)
        print("Classification score %s (pvalue : %s)" % (score, pvalue))


#trainW2v()

if __name__ == "__main__":
    parser = OptionParser('''%prog -o ontology -t type -f force ''')
    parser.add_option('-o', '--ontology', dest='ontology', default="yago")
    parser.add_option('-t', '--type', dest='type', default="specific")
    parser.add_option('-f', '--force', dest='force', default=0, type=int)
    parser.add_option('-c', '--classifier', dest='classifier', default='ben')
    parser.add_option('-j', '--job', dest='job', type=int, default=10)
    parser.add_option('-w', '--window', dest='window', type=int, default=2)
    parser.add_option('-s', '--size', dest='size', type=int, default=300)
    parser.add_option('-m', '--merge', dest='merge', type=int, default=0)
    parser.add_option('-e', '--experiment', dest='experiment', type=int, default=1)
    opts, args = parser.parse_args()

    trainW2v(opts)