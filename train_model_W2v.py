from sklearn.naive_bayes import *
from sklearn import svm
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix
from optparse import OptionParser
from helper import FileHelper, Word2VecHelper, GraphHelper
import helper
from helper.VectorHelper import *
classes = []

import os
import sys


#log = helper.enableLog()

def trainW2v(args):
    global classes
    clazz = [["Accidents", "Arts", "Attacks", "Economy", "Miscellaneous", "Politics", "Science", "undefined"], ["Accidents", "Arts", "Attacks", "Economy", "Miscellaneous", "Politics", "Science"]]

    FileHelper.create("logs")

    C = 2.0  # SVM regularization parameter
    gamma = 0.5
    degree = 6
    types = ['generic', 'specific']
    if args.ontology =='dbpedia':
        types.append('normal')

    for classes in clazz:
        task = 'pipeline2' if len(clazz) == 8 else 'task2' if len(clazz) == 7 else 'task1'
        train_instances, train_labels, train_texts = Word2VecHelper.loadData(classes, args, 'train')
        test_instances, test_labels, test_texts = Word2VecHelper.loadData(classes, args, 'test')


        f = open(
           "logs/{}_{}.txt".format(args.ontology, task), "w")

        sys.stdout = f

        for classifier in ['ben', 'linear', 'rbf']:
            args.classifier = classifier

            for _type in types:
                args.type = _type
                for merge in range(2):
                    args.merge = merge
                    if args.force == 1 or not os.path.exists("{}_{}_{}.bin".format(args.ontology, args.type, 'merged' if args.merge==1 else 'simple')):
                        files = ["./train/{}/{}/positive.txt".format(args.ontology, args.type),
                                 "./train/{}/{}/negative.txt".format(args.ontology, args.type)]
                        model = Word2VecHelper.createModel(files, name="{}_{}".format(args.ontology, args.type),
                                                           merge=args.merge)
                    else:
                        model = Word2VecHelper.loadModel("{}_{}".format(args.ontology, args.type), merge=args.merge)

                    w2v = {w: vec for w, vec in zip(model.wv.index2word, model.wv.syn0)}

                    print("========== Model", args.ontology, args.type, args.merge, task, classifier, "==========")
                    if args.classifier == 'ben':
                        classifier = Pipeline([("w2v vect", MeanEmbeddingVectorizer(w2v)),
                                               ("clf", BernoulliNB())])
                    else:
                        classifier = Pipeline([("w2v vect", MeanEmbeddingVectorizer(w2v)),
                                               ("clf", svm.SVC(kernel=args.classifier, degree=degree, C=C, gamma=gamma,
                                                                       probability=True))])

                    y_score = classifier.fit(train_texts, train_labels).predict_proba(test_texts)
                    y_pred = classifier.predict(test_texts)
                    #f.write("========= Classification Report ==========\n")
                    print("========= Classification Report ==========")
                    print(classification_report(test_labels, y_pred))
                    #f.write(classification_report(test_labels, y_pred)+"\n")

                    print("========= Confusion Matrix ==========")
                    #f.write("========= Confusion Matrix ==========\n")
                    print(confusion_matrix(test_labels,y_pred, labels=classes))
                    #f.write(confusion_matrix(test_labels,y_pred, labels=classes)+"\n")

                    GraphHelper.savePrediction("{}_{}_{}_{}_{}".format(args.ontology,args.type,args.classifier,task, args.merge), y_pred=y_pred,y_score=y_score,classes=classes,y=test_labels )
                    GraphHelper.saveClassifier(classifier, "{}_{}_{}_{}_{}.pkl".format(args.ontology,args.type,args.classifier,task, args.merge))
                    break
                break
            break
        break

        #f.close()

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