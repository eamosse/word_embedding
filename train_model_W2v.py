from sklearn.naive_bayes import *
from sklearn import svm
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix
from optparse import OptionParser
from helper import FileHelper, Word2VecHelper
import helper
from helper.VectorHelper import *
classes = []


log = helper.enableLog()


def trainW2v(args):
    global classes
    if args.experiment == 1:
        classes = ["Accidents", "Arts", "Attacks", "Economy", "Miscellaneous", "Politics", "Science"]
    else:
        classes = ['negative', 'positive']
    if args.force == 1 :
        log.debug("Generatiing files....")
        FileHelper.generateDataFile()
        log.debug("Building the W2V model")
        files = ["./train/{}/{}/positive.txt".format(args.ontology, args.type),
                "./train/{}/{}/negative.txt".format(args.ontology, args.type)]
        model = Word2VecHelper.createModel(name="{}_{}".format(args.ontology, args.type), files=files)
    else:
        #model = Word2VecHelper.loadModel("GoogleNews-vectors-negative300")
        model = Word2VecHelper.loadModel("{}_{}".format(args.ontology,args.type))

    w2v = {w: vec for w, vec in zip(model.wv.index2word, model.wv.syn0)}


    train_instances, train_labels, train_texts = Word2VecHelper.loadData(classes,args, 'train')
    test_instances, test_labels, test_texts = Word2VecHelper.loadData(classes,args, 'test')

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
    classifier_count.fit(train_texts, train_labels)
    y_pred = classifier_count.predict(test_texts)

    print(classification_report(test_labels, y_pred))
    print(confusion_matrix(test_labels,y_pred, labels=classes))



    """
    log.debug("BUilding the classifier {} with tfidf".format(args.classifier))
    classifier_tfidf.fit(x_train, y_train)
    y_pred = classifier_tfidf.predict(x_test)
    print(y_pred)
    print(classification_report(y_test, y_pred))
    print(confusion_matrix(y_test, y_pred, labels=classes))
    """

    #print((cross_val_score(etree_w2v, X, y, cv=10).mean()))


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
    parser.add_option('-m', '--min', dest='min_count', type=int, default=5)
    parser.add_option('-e', '--experiment', dest='experiment', type=int, default=1)
    opts, args = parser.parse_args()

    trainW2v(opts)