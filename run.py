import numpy as np
GLOVE_6B_200D_PATH = "glove.twitter.27B/glove.twitter.27B.200d.txt"

with open(GLOVE_6B_200D_PATH, "r") as lines:
     word2vec = {line.split()[0]: np.array([float(i) for i in line.split()[1:]])
                 for line in lines}

for k in word2vec.keys():
    print(k)