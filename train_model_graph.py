from sklearn.naive_bayes import *
from sklearn import svm
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix
from optparse import OptionParser
from helper import FileHelper, Word2VecHelper
import helper
import numpy as np
from helper.VectorHelper import *
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
import matplotlib.pyplot as plt
classes = []
from itertools import cycle
from sklearn.preprocessing import label_binarize



log = helper.enableLog()


def trainW2v(args):
    global classes
    if args.experiment == 1:
        classes = ["Accidents", "Arts", "Attacks", "Economy", "Miscellaneous", "Politics", "Science"]
    else:
        classes = ['negative', 'positive']

    if args.force == 1:
        files = ["./train/{}/{}/positive.txt".format(args.ontology, args.type),
                 "./train/{}/{}/negative.txt".format(args.ontology, args.type)]
        model = Word2VecHelper.createModel(files, name="{}_{}".format(args.ontology, args.type), merge=args.merge)
    else:
        model = Word2VecHelper.loadModel("{}_{}".format(args.ontology, args.type), merge=args.merge)

    w2v = {w: vec for w, vec in zip(model.wv.index2word, model.wv.syn0)}

    train_instances, train_labels, train_texts = Word2VecHelper.loadData(classes,args, 'train')
    test_instances, test_labels, test_texts = Word2VecHelper.loadData(classes,args, 'test')

    #test_labels = label_binarize(test_labels, classes=classes)
    #train_labels = label_binarize(train_labels, classes=classes)
    #n_classes = len(classes) #test_labels.shape[1]
    y = label_binarize(test_labels, classes=classes)
    print(y)
    n_classes = y.shape[1]

    C = 2.0  # SVM regularization parameter
    gamma = 10
    degree = 6
    colors = cycle(['navy', 'turquoise', 'darkorange', 'cornflowerblue', 'teal'])
    lw = 2

    for cl in ['linear']:
        args.classifier = cl
        print("RUNNING ", cl)

        if args.classifier == 'poly':
            classifier_count = Pipeline([("word2vec vectorizer", MeanEmbeddingVectorizer(w2v)),
                              ("extra trees", svm.SVC(kernel="poly", degree=degree, C=C, gamma=gamma, probability=True))])

            classifier_tfidf = Pipeline([("word2vec vectorizer", TfidfEmbeddingVectorizer(w2v)),
                              ("extra trees", svm.SVC(kernel="poly", degree=3,C=C))])
        elif args.classifier == 'linear':
            classifier_count = Pipeline([("word2vec vectorizer", MeanEmbeddingVectorizer(w2v)),
                                         ("extra trees", svm.SVC(kernel="linear", C=C, probability=True))])

            classifier_tfidf = Pipeline([("word2vec vectorizer", TfidfEmbeddingVectorizer(w2v)),
                                         ("extra trees", svm.SVC(kernel="linear", C=C, probability=True))])
        elif args.classifier == 'rbf':
            classifier_count = Pipeline([("word2vec vectorizer", MeanEmbeddingVectorizer(w2v)),
                                         ("extra trees", svm.SVC(kernel='rbf', gamma=10, C=C, probability=True))])

            classifier_tfidf = Pipeline([("word2vec vectorizer", TfidfEmbeddingVectorizer(w2v)),
                                         ("extra trees", svm.SVC(kernel='rbf', gamma=10, C=C, probability=True))])
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
        y_score = classifier_count.fit(train_texts, train_labels).predict_proba(test_texts)

        y_pred = classifier_count.predict(test_texts)

        print(classification_report(test_labels, y_pred))
        print(confusion_matrix(test_labels,y_pred, labels=classes))
        """
        f = open('{}_{}_{}_{}.tsv'.format(args.ontology,args.type,args.classifier,args.merge), "w")
        t = ""
        for cl in classes:
            t =  "\t" + cl if len(t) == 0 else cl
            f.write(t)
        """

        #np.savetxt("test.csv", y_score, delimiter = ';')

        # Compute Precision-Recall and plot curve
        precision = dict()
        recall = dict()
        average_precision = dict()
        #print(n_classes, test_labels.shape[1])
        #print(n_classes, y_pred.shape[1])
        for i in range(n_classes):
            precision[i], recall[i], _ = precision_recall_curve(y[:, i],
                                                                y_score[:, i])
            average_precision[i] = average_precision_score(y[:, i], y_score[:, i])

        # Compute micro-average ROC curve and ROC area
        precision["micro"], recall["micro"], _ = precision_recall_curve(y.ravel(),
                                                                        y_score.ravel())
        average_precision["micro"] = average_precision_score(y, y_score,
                                                             average="micro")


        # Plot Precision-Recall curve
        plt.clf()
        plt.plot(recall[0], precision[0], lw=lw, color='navy',
                 label='Precision-Recall curve')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.ylim([0.0, 1.05])
        plt.xlim([0.0, 1.0])
        plt.title('Precision-Recall Curve: AUC={0:0.2f}'.format(average_precision[0]))
        plt.legend(loc="upper right")
        plt.savefig('{}_{}_{}_{}_prc.svg'.format(args.ontology, args.type, args.classifier, args.merge), format="svg",
                    facecolor='w', edgecolor='w',
                    orientation='portrait', papertype=None,
                    transparent=False, bbox_inches=None, pad_inches=0.1,
                    frameon=None)
        #plt.show()


        # Plot Precision-Recall curve for each class
        plt.clf()
        plt.plot(recall["micro"], precision["micro"], color='gold', lw=lw,
                 label='micro-average Precision-recall curve (area = {0:0.2f})'
                       ''.format(average_precision["micro"]))
        for i, color in zip(range(n_classes), colors):
            plt.plot(recall[i], precision[i], color=color, lw=lw,
                     label='{0} (area = {1:0.2f})'
                           ''.format(classes[i], average_precision[i]))

        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall curve for model {} {}'.format(args.ontology, args.type))
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.05),
          ncol=3, fancybox=True)
        plt.savefig('{}_{}_{}_{}_average.svg'.format(args.ontology, args.type,args.classifier,args.merge), format="svg", facecolor='w', edgecolor='w',
        orientation='portrait', papertype=None,
        transparent=False, bbox_inches=None, pad_inches=0.1,
        frameon=None)

        from helper import GraphHelper
        GraphHelper.savePrediction("test.npz",y_score,y_pred,test_labels,classes)


        #plt.show()



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
    parser.add_option('-o', '--ontology', dest='ontology', default="dbpedia")
    parser.add_option('-t', '--type', dest='type', default="generic")
    parser.add_option('-f', '--force', dest='force', default=0, type=int)
    parser.add_option('-c', '--classifier', dest='classifier', default='poly')
    parser.add_option('-j', '--job', dest='job', type=int, default=10)
    parser.add_option('-w', '--window', dest='window', type=int, default=2)
    parser.add_option('-s', '--size', dest='size', type=int, default=300)
    parser.add_option('-m', '--merge', dest='merge', type=int, default=0)
    parser.add_option('-e', '--experiment', dest='experiment', type=int, default=1)
    opts, args = parser.parse_args()

    trainW2v(opts)