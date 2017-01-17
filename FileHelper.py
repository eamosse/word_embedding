from helper import MongoHelper as db

db.connect("tweets_dataset")

def parse(data):
    texts = []
    for t in data:
        text = t['text_snowball']
        text = ' '.join([t for t in text.split() if len(t) > 2])
        if len(text) > 0:
            texts.append(text)
    return texts

def write(data, file):
    with open(file, "w") as f:
        texts = parse(data)
        #print(texts)
        f.write('\n'.join(texts))

def generate(type,ontology):
    data = db.find("annotated", query={"type":type, "ontology":ontology, "dataset":"event 2012", "category":{"$ne":"undefined"}})
    write(data=data,file='train_positive.txt')

    data = db.find("annotated", query={"type":type, "ontology":ontology, "dataset":"event 2012", "category":"undefined"})
    write(data=data, file='train_negative.txt')

    data = db.find("annotated", query={"type":type, "ontology":ontology, "dataset":"fsd", "category":{"$ne":"undefined"}})
    write(data=data, file='test_positive.txt')

    data = db.find("annotated", query={"type":type, "ontology":ontology, "dataset":"fsd", "category":"undefined"})
    write(data=data, file='test_negative.txt')


def nbLines(file):
    num_lines = sum(1 for line in open(file))
    return num_lines

#generate("normal","dbpedia")