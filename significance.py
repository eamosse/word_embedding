from helper import  GraphHelper, FileHelper
import numpy as np
import sys
def mapping(prediction, target):
    return [1 if prediction[i] == target[i] else 0 for i in range(len(prediction))]


if __name__ == '__main__':
    FileHelper.create("single_pipeline")

    clazz = ["Accidents", "Arts", "Attacks", "Economy", "Miscellaneous", "Politics", "Science", "Sports"]
    ontologies = ["yago","dbpedia"]
    settings = ["generic","specific"]
    models=["linear", "ben"]
    pipelines = ['pipeline1', 'pipeline2']
    sys.stdout = open("test.log", "w")

    for ontology in ontologies:
        if ontology=='dbpedia':
            settings.append("normal")
        for setting in settings:
            for model in models:
                for w in range(2):
                    model1 = '{}_{}_{}_{}_{}.npz'.format(ontology,setting,model,pipelines[0],w)
                    model2 = '{}_{}_{}_{}_{}.npz'.format(ontology,setting,model,pipelines[1],w)
                    ids1, labels1, preds1 = GraphHelper.loadPredictionStat(model1)
                    ids2, labels2, preds2 = GraphHelper.loadPredictionStat(model2)
                    val1, val2 = [],[]
                    mval = {'val1':[], 'val2':[]}

                    a1 = mapping(labels1, preds1)
                    a2 = mapping(labels2, preds2)
                    for index2, _id in enumerate(ids2):
                        if not _id in ids1 or not labels2[index2] in clazz:
                            continue
                        index1 = list(ids1).index(_id)
                        print(_id, a2[index2],a1[index1])
                        mval['val2'].append(a2[index2])
                        mval['val1'].append(a1[index1])

                    arr = np.array([mval['val1'], mval['val2']])
                    arr = arr.transpose()
                    np.savetxt("single_pipeline/{}_{}_{}_{}.txt".format("SingleVSPipeline", "NB" if model=='ben' else 'SVM', ontology,setting), arr,
                               delimiter="\t", fmt="%d")
                    break
                break
            break
        break

