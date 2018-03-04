import os

current_filepath = os.path.dirname(os.path.abspath(__file__))
ROOT_PATH = os.path.abspath(os.path.join(current_filepath, os.pardir))

TRAIN_PATH = os.path.join(ROOT_PATH, "data/processed/train.csv")
TEST_PATH = os.path.join(ROOT_PATH, "data/processed/test.csv")
GLOVE_PATH = os.path.join(ROOT_PATH, "data/external/glove.840B.300d.txt")
WORD2VEC_PATH = os.path.join(ROOT_PATH, "data/external/word2vec.300.bin")
