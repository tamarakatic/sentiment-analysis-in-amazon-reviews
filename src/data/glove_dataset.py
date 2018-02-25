import numpy as np


def glove_dataset(path):
    print("\nLoading GloVe dataset...")
    glove_w2v = {}
    with open(path, 'rb') as f:
        for line in f:
            if line != 0:
                values = line.split()
                word = values[0].decode("utf-8")
                coefs = np.asarray(values[1:], dtype='float32')
                glove_w2v[word] = coefs
    return glove_w2v
