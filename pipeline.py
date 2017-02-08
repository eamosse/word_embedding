from sklearn.naive_bayes import *
from sklearn import svm
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix
from optparse import OptionParser
from helper import FileHelper, Word2VecHelper, GraphHelper
import helper
from sklearn.externals import joblib
from helper.VectorHelper import *
import sys

log = helper.enableLog()


def trainW2v(args):
    all = ["Accidents", "Arts", "Attacks", "Economy", "Miscellaneous", "Politics", "Science", "Sports"]
    pipeline = 'pipeline2' if len(all) ==9 else 'task2'
    binaries = ['negative', 'positive']
    models = ['ben', 'linear']
    types = ['generic', 'specific']
    test_instances_binary, test_labels_binary, test_texts_binary = Word2VecHelper.loadData(binaries, args, 'test')
    if args.ontology == 'dbpedia':
        types.append('normal')
    #task = 'pipeline2' if len(classes) == 9 else 'task2' if len(classes) == 8 else 'task1'
    sys.stdout = open(
        "log/{}_pipeline1_{}.txt".format(args.ontology, len(all)), "w")
    for model in models:
        args.classifier = model
        for _type in types:
            args.type = _type
            for merge in range(2):
                args.merge = merge
                log.debug("Loading the binary model".format(args.classifier))
                binary_model = GraphHelper.loadClassifier("{}_{}_{}_{}_{}.pkl".format(args.ontology, args.type, args.classifier, "task1", args.merge))
                log.debug("Loading the multi model".format(args.classifier))
                multi_model = GraphHelper.loadClassifier("{}_{}_{}_{}_{}.pkl".format(args.ontology, args.type, args.classifier, pipeline, args.merge))
                #x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

                y_pred = binary_model.predict(test_texts_binary)

                #build the training and test file for task 2
                ids = []
                for i, label in enumerate(y_pred):
                    if label == 'positive':
                        ids.append(test_instances_binary[i])
                #this file contains tweets classified as related to events by the binary model
                eval_file = FileHelper.generateFileForIds(ids=ids, ontology=args.ontology, type=args.type)
                #we then use the positive tweets as instances for testing the multi model
                test_ids, test_labels_multi, test_texts_multi = Word2VecHelper.dataFromFile(eval_file)
                #print("==========","Model for", args.ontology, args.type, args.classifier, args.merge,"==========")
                #Train the multi class model
                y_pred = multi_model.predict(test_texts_multi)
                #print(classification_report(test_labels_multi, y_pred))
                #print(confusion_matrix(test_labels_multi, y_pred, labels=all))
                y_score = multi_model.predict_proba(test_texts_multi)
                GraphHelper.savePredictionForStatistics(
                    "{}_{}_{}_{}_{}".format(args.ontology, args.type, args.classifier, "pipeline1", args.merge), test_ids,test_labels_multi,y_pred)


if __name__ == "__main__":
    parser = OptionParser('''%prog -o ontology -t type -f force ''')
    parser.add_option('-o', '--ontology', dest='ontology', default="yago")
    parser.add_option('-t', '--type', dest='type', default="specific")
    parser.add_option('-f', '--force', dest='force', default=1, type=int)
    parser.add_option('-c', '--classifier', dest='classifier', default='ben')
    parser.add_option('-j', '--job', dest='job', type=int, default=10)
    parser.add_option('-w', '--window', dest='window', type=int, default=2)
    parser.add_option('-s', '--size', dest='size', type=int, default=300)
    parser.add_option('-m', '--merge', dest='merge', type=int, default=0)
    parser.add_option('-e', '--experiment', dest='experiment', type=int, default=1)
    opts, args = parser.parse_args()

    trainW2v(opts)