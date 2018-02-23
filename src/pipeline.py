from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from mean_embedding_vectorizer import MeanEmbeddingVectorizer
from data.glove_dataset import glove_dataset
from tfidf_embedding_vectorizer import TfidfEmbeddingVectorizer

glove_w2v = glove_dataset('../dataset/glove.840B.300d.txt')


def bag_of_words(classifier, tf_idf=True):
    if tf_idf:
        steps = [('vect', TfidfVectorizer())]
    else:
        steps = [('vect', CountVectorizer())]

    steps.append(('cls', classifier))
    return Pipeline(steps)


def glove_vectorizer(classifier):
    steps = [('vect', MeanEmbeddingVectorizer(glove_w2v))]
    steps.append(('cls', classifier))
    return Pipeline(steps)


def tfidf_vectorizer(classifier):
    steps = [('vect', TfidfEmbeddingVectorizer(glove_w2v))]
    steps.append(('cls', classifier))
    return Pipeline(steps)
