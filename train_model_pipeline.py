from sklearn.naive_bayes import *
from sklearn import svm
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix
from optparse import OptionParser
from helper import FileHelper, Word2VecHelper
import helper
from helper.VectorHelper import *

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

    cl = ["ben","linear"]

    for cc in cl:
        args.classifier = cc
        print("RUNNING FOR ", cc)
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
        log.debug("Train the binary model".format(args.classifier))
        classifier_count.fit(train_texts, train_labels)
        y_pred = classifier_count.predict(test_texts)

        print(classification_report(test_labels, y_pred))
        print(confusion_matrix(test_labels,y_pred, labels=binaries))


        #build the training and test file for task 2
        ids = []
        for i, label in enumerate(y_pred):
            if label == 'positive':
                ids.append(test_instances[i])
        eval_file = FileHelper.generateFileForIds(ids=ids, ontology=args.ontology, type=args.type)

        train_instances_multi, train_labels_multi, train_texts_multi = Word2VecHelper.loadData(all, args, 'train')
        _, test_labels_multi, test_texts_multi = Word2VecHelper.dataFromFile(eval_file)

        #Train the multi class model
        classifier_count.fit(train_texts_multi, train_labels_multi)
        y_pred = classifier_count.predict(test_texts_multi)
        print(classification_report(test_labels_multi, y_pred))
        print(confusion_matrix(test_labels_multi, y_pred, labels=all))



#trainW2v()

if __name__ == "__main__":
    parser = OptionParser('''%prog -o ontology -t type -f force ''')
    parser.add_option('-o', '--ontology', dest='ontology', default="yago")
    parser.add_option('-t', '--type', dest='type', default="specific")
    parser.add_option('-f', '--force', dest='force', default=1, type=int)
    parser.add_option('-c', '--classifier', dest='classifier', default='ben')
    parser.add_option('-j', '--job', dest='job', type=int, default=10)
    parser.add_option('-w', '--window', dest='window', type=int, default=2)
    parser.add_option('-s', '--size', dest='size', type=int, default=300)
    parser.add_option('-m', '--merge', dest='merge', type=bool, default=False)
    parser.add_option('-e', '--experiment', dest='experiment', type=int, default=1)
    opts, args = parser.parse_args()

    trainW2v(opts)