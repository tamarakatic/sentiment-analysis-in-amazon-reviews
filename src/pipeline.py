import os

from definitions import ROOT_PATH
from definitions import GLOVE_PATH

from gensim.models import Doc2Vec

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

from word_embedding import MeanEmbeddingVectorizer
from word_embedding import TfidfEmbeddingVectorizer
from word_embedding import Doc2VecVectorizer
from word_embedding.embedding_loader import loading_embedding_dataset


def bag_of_words(classifier, tf_idf=True):
    if tf_idf:
        steps = [('vect', TfidfVectorizer())]
    else:
        steps = [('count', CountVectorizer())]

    steps.append(('cls', classifier))
    return Pipeline(steps)


def glove_mean_vectorizer(classifier, word2vec=None):
    if word2vec:
        embedding = word2vec
    else:
        embedding = loading_embedding_dataset(GLOVE_PATH)

    steps = [('vect', MeanEmbeddingVectorizer(embedding))]
    steps.append(('cls', classifier))
    return Pipeline(steps)


def glove_tfidf_vectorizer(classifier, word2vec=None):
    if word2vec:
        embedding = word2vec
    else:
        embedding = loading_embedding_dataset(GLOVE_PATH)

    steps = [('vect', TfidfEmbeddingVectorizer(embedding))]
    steps.append(('cls', classifier))
    return Pipeline(steps)


def doc2vec(classifier):
    model_filename = "models/doc2vec.model"
    model_path = os.path.join(ROOT_PATH, model_filename)
    model = Doc2Vec.load(model_path)

    steps = [
        ('vect', Doc2VecVectorizer(model)),
        ('cls', classifier)
    ]

    return Pipeline(steps)
