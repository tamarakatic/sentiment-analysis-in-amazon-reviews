import os

from definitions import ROOT_PATH
from definitions import GLOVE_PATH
from definitions import WORD2VEC_PATH

from gensim.models import Doc2Vec

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from data.loaders import load_word2vec_embedding_matrix
from data.loaders import load_glove_embedding_matrix

from vectorizers import MeanEmbeddingVectorizer
from vectorizers import TfidfMeanEmbeddingVectorizer
from vectorizers import Doc2VecVectorizer


def bag_of_words(classifier, tf_idf=False):
    if tf_idf:
        steps = [("vect", TfidfVectorizer())]
    else:
        steps = [("vect", CountVectorizer())]

    steps.append(("cls", classifier))
    return Pipeline(steps)


def doc2vec(classifier=LogisticRegression()):
    model_filename = "models/doc2vec.model"
    model_path = os.path.join(ROOT_PATH, model_filename)
    model = Doc2Vec.load(model_path)

    steps = [
        ("vect", Doc2VecVectorizer(model)),
        ("cls", classifier)
    ]

    return Pipeline(steps)


def glove_mean_embedding(classifier=LogisticRegression(), tf_idf=False):
    embedding_matrix = load_glove_embedding_matrix(GLOVE_PATH)
    return mean_embedding(classifier, embedding_matrix, tf_idf)


def word2vec_mean_embedding(classifier=LogisticRegression(), tf_idf=False):
    embedding_matrix = load_word2vec_embedding_matrix(WORD2VEC_PATH)
    return mean_embedding(classifier, embedding_matrix, tf_idf)


def mean_embedding(classifier, embedding_matrix, tf_idf=False):
    if tf_idf:
        steps = [("vect", TfidfMeanEmbeddingVectorizer(embedding_matrix))]
    else:
        steps = [("vect", MeanEmbeddingVectorizer(embedding_matrix))]

    steps.append(("cls", classifier))
    return Pipeline(steps)
