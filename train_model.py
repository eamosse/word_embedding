# gensim modules
from gensim import utils
from gensim.models.doc2vec import TaggedDocument
from gensim.models import Doc2Vec
import FileHelper
from optparse import OptionParser

# random shuffle
from random import shuffle
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix

# numpy
import numpy

# classifier
from sklearn import svm
from sklearn.naive_bayes import  GaussianNB
from sklearn.linear_model import LogisticRegression

import logging
import sys

log = logging.getLogger()
log.setLevel(logging.DEBUG)

ch = logging.StreamHandler(sys.stdout)
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)
log.addHandler(ch)



class TaggedLineSentence(object):
    def __init__(self, sources):
        self.sources = sources

        flipped = {}

        # make sure that keys are unique
        for key, value in sources.items():
            if value not in flipped:
                flipped[value] = [key]
            else:
                raise Exception('Non-unique prefix encountered')

    def __iter__(self):
        for source, prefix in self.sources.items():
            with utils.smart_open(source) as fin:
                for item_no, line in enumerate(fin):
                    yield TaggedDocument(utils.to_unicode(line).split(), [prefix + '_%s' % item_no])

    def to_array(self):
        self.sentences = []
        for source, prefix in self.sources.items():
            with utils.smart_open(source) as fin:
                for item_no, line in enumerate(fin):
                    self.sentences.append(TaggedDocument(utils.to_unicode(line).split(), [prefix + '_%s' % item_no]))
        return self.sentences

    def sentences_perm(self):
        shuffle(self.sentences)
        return self.sentences

def createModel(sources, name):
    log.info('TaggedDocument')
    sentences = TaggedLineSentence(sources)

    log.info('D2V')
    model = Doc2Vec(min_count=1, window=10, size=100, sample=1e-4, negative=5, workers=7)
    model.build_vocab(sentences.to_array())

    log.info('Epoch')
    for epoch in range(10):
        log.info('EPOCH: {}'.format(epoch))
        model.train(sentences.sentences_perm())

    log.info('Model Save')
    model.save('./{}.d2v'.format(name))
    return model

def train(args):
    if args.force == 1:
        log.info('generating the files...')
        FileHelper.generate(type=args.type, ontology=args.ontology)
        model_train = createModel(sources={'train_positive.txt':'TRAIN_POS', 'train_negative.txt':'TRAIN_NEG'},
                                  name="train_{}_{}".format(args.type,args.ontology))
        createModel(sources={'test_negative.txt':'TEST_NEG', 'test_positive.txt':'TEST_POS'}, name="test_{}_{}".format(args.type,args.ontology))
    else:
        model_train = Doc2Vec.load('./train_{}_{}.d2v'.format(args.type, args.ontology))

    log.info('Event Classification')
    train_positive_length = FileHelper.nbLines('train_positive.txt')
    train_negative_length = FileHelper.nbLines('train_negative.txt')
    train_arrays = numpy.zeros((train_positive_length+train_negative_length, 100))
    train_labels = numpy.zeros(train_positive_length+train_negative_length)

    for i in range(train_positive_length):
        prefix_train_pos = 'TRAIN_POS_' + str(i)
        train_arrays[i] = model_train.docvecs[prefix_train_pos]
        train_labels[i] = 1

    for i in range(train_positive_length, train_positive_length+train_negative_length):
        prefix_train_neg = 'TRAIN_NEG_' + str(i-train_positive_length)
        train_arrays[i] = model_train.docvecs[prefix_train_neg]
        train_labels[i] = 0

    C = 1.0  # SVM regularization parameter
    log.info('Fitting the model')
    if args.classifier =="svc":
        classifier = svm.SVC(kernel='linear', C=C)
    elif args.classifier == "rbf":
        classifier = svm.SVC(kernel='rbf', gamma=0.7, C=C)
    elif args.classifier == "poly":
        classifier = svm.SVC(kernel='poly', degree=3, C=C)
    elif args.classifier == "linear":
        classifier = svm.LinearSVC(C=C)
    elif args.args == "nb" :
        classifier = GaussianNB()
    else:
        classifier = LogisticRegression()

    classifier.fit(train_arrays, train_labels)
    test(classifier=classifier,args=args)

def test(classifier,args):
    model_test = Doc2Vec.load('./test_{}_{}.d2v'.format(args.type, args.ontology))
    test_positive_length = FileHelper.nbLines('test_positive.txt')
    test_negative_length = FileHelper.nbLines('test_negative.txt')

    total = test_positive_length + test_negative_length

    test_arrays = numpy.zeros((total, 100))
    test_labels = numpy.zeros(total)

    for i in range(test_positive_length):
        prefix_test_pos = 'TEST_POS_' + str(i)
        test_arrays[i] = model_test.docvecs[prefix_test_pos]
        test_labels[i] = 1

    for i in range(test_positive_length, total):
        prefix_test_neg = 'TEST_NEG_' + str(i - test_positive_length)
        test_arrays[i] = model_test.docvecs[prefix_test_neg]
        test_labels[i] = 0

    log.info('Evaluation....')
    y_pred = classifier.predict(test_arrays)
    log.info("Ontology:{} - Type : {} ".format(args.ontology, args.type))

    print("Precision",precision_score(test_labels, y_pred, average="weighted"))
    print("Recall",recall_score(test_labels, y_pred, average="weighted", labels=[1,0]))
    print("F1", f1_score(test_labels, y_pred, average="weighted", labels=[1,0]))

    #print(classifier.score(test_arrays, test_labels))

#train()

if __name__ == "__main__":
    parser = OptionParser('''%prog -o ontology -t type -f force ''')
    parser.add_option('-o', '--ontology', dest='ontology', default="dbpedia")
    parser.add_option('-t', '--type', dest='type', default="generic")
    parser.add_option('-f', '--force', dest='force', default=0)
    parser.add_option('-c', '--classifier', dest='classifier', default='poly')
    opts, args = parser.parse_args()
    train(opts)