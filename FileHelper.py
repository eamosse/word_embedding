from helper import MongoHelper as db

db.connect("tweets_dataset")

def generate(type,ontology):
    #sources = {'test_negative.txt': 'TEST_NEG', 'test_positive.txt': 'TEST_POS', 'train_negative.txt': 'TRAIN_NEG',
               #'train_positive.txt': 'TRAIN_POS'}
    data = db.find("annotated", query={"type":type, "ontology":ontology, "dataset":"event 2012", "category":{"$ne":"undefined"}})
    with open("train_positive.txt", "w") as f:
        texts = [t['text_snowball'] for t in data]
        f.write('\n'.join(texts))

    data = db.find("annotated", query={"type":type, "ontology":ontology, "dataset":"event 2012", "category":"undefined"})
    with open("train_negative.txt", "w") as f:
        texts = [t['text_snowball'] for t in data]
        f.write('\n'.join(texts))


    data = db.find("annotated", query={"type":type, "ontology":ontology, "dataset":"fsd", "category":{"$ne":"undefined"}})
    with open("test_positive.txt", "w") as f:
        texts = [t['text_snowball'] for t in data]
        f.write('\n'.join(texts))


    data = db.find("annotated", query={"type":type, "ontology":ontology, "dataset":"fsd", "category":"undefined"})
    with open("test_negative.txt", "w") as f:
        texts = [t['text_snowball'] for t in data]
        f.write('\n'.join(texts))


def nbLines(file):
    num_lines = sum(1 for line in open(file))
    return num_lines


