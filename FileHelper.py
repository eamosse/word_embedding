from helper import MongoHelper as db
import os
db.connect("tweets_dataset")

def parse(data):
    texts = []
    for t in data:
        text = t['text']
        text = ' '.join([t for t in text.split() if len(t) > 2])
        if len(text) > 0:
            texts.append(text)
    return texts

def write(data, file):
    with open(file, "w", encoding='utf-8') as f:
        texts = parse(data)
        #print(texts)
        f.write('\n'.join(texts).strip())

def generate(type,ontology):
    if not os.path.exists("train"):
        os.makedirs("train")
    if not os.path.exists("test"):
        os.makedirs("test")

    data = db.find("annotated", query={"type":type, "ontology":ontology, "dataset":"event 2012", "category":{"$ne":"undefined"}})
    write(data=data,file='train/positive.txt')

    data = db.find("annotated", query={"type":type, "ontology":ontology, "dataset":"event 2012", "category":"undefined"})
    write(data=data, file='train/negative.txt')

    data = db.find("annotated", query={"type":type, "ontology":ontology, "dataset":"fsd", "category":{"$ne":"undefined"}})
    write(data=data, file='test/positive.txt')

    data = db.find("annotated", query={"type":type, "ontology":ontology, "dataset":"fsd", "category":"undefined"})
    write(data=data, file='test/negative.txt')


def nbLines(file):
    num_lines = sum(1 for line in open(file))
    return num_lines

#generate("generic","dbpedia")