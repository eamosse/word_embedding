# gensim modules
from gensim import utils
from gensim.models.doc2vec import TaggedDocument
from gensim.models import Doc2Vec
import FileHelper

# random shuffle
from random import shuffle
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix

# numpy
import numpy

# classifier
from sklearn import svm

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

def createModel(type,ontology):
    log.info('generating the files...')
    FileHelper.generate(type=type, ontology=ontology)
    log.info('source load')
    sources = {'test_negative.txt':'TEST_NEG', 'test_positive.txt':'TEST_POS', 'train_negative.txt':'TRAIN_NEG', 'train_positive.txt':'TRAIN_POS'}

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
    model.save('./{}_{}.d2v'.format(ontology,type))

def train(type,ontology,force=False):
    if force:
        createModel(type=type,ontology=ontology)
    model = Doc2Vec.load('./{}_{}.d2v'.format(ontology,type))
    log.info('Sentiment')

    train_positive_length = FileHelper.nbLines('train_positive.txt')
    train_negative_length = FileHelper.nbLines('train_negative.txt')
    test_positive_length = FileHelper.nbLines('test_positive.txt')
    test_negative_length = FileHelper.nbLines('test_negative.txt')

    train_arrays = numpy.zeros((train_positive_length+train_negative_length, 100))
    train_labels = numpy.zeros(train_positive_length+train_negative_length)

    for i in range(train_positive_length):
        prefix_train_pos = 'TRAIN_POS_' + str(i)
        train_arrays[i] = model.docvecs[prefix_train_pos]
        train_labels[i] = 1

    for i in range(train_positive_length, train_positive_length+train_negative_length):
        prefix_train_neg = 'TRAIN_NEG_' + str(i-train_positive_length)
        train_arrays[i] = model.docvecs[prefix_train_neg]
        train_labels[i] = 0

    total = test_positive_length+test_negative_length

    test_arrays = numpy.zeros((total, 100))
    test_labels = numpy.zeros(total)

    for i in range(test_positive_length):
        prefix_test_pos = 'TEST_POS_' + str(i)
        test_arrays[i] = model.docvecs[prefix_test_pos]
        test_labels[i] = 1

    for i in range(test_positive_length,total):
        prefix_test_neg = 'TEST_NEG_' + str(i-test_positive_length)
        test_arrays[i] = model.docvecs[prefix_test_neg]
        test_labels[i] = 0

    log.info('Fitting')
    classifier = svm.SVC()
    #classifier = LogisticRegression()
    classifier.fit(train_arrays, train_labels)

    #LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
              #intercept_scaling=1, penalty='l2', random_state=None, tol=0.0001)

    #y_pred = classifier.predict(test_arrays)
    #print(f1_score(y_test, y_pred, average="macro"))
    #print(precision_score(y_test, y_pred, average="macro"))
    #print(recall_score(y_test, y_pred, average="macro"))

    log.info('Evaluation....')

    print(classifier.score(test_arrays, test_labels))

train(type='generic',ontology='dbpedia',force=False)