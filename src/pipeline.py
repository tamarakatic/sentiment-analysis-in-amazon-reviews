from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from mean_embedding_vectorizer import MeanEmbeddingVectorizer
from data.glove_dataset import glove_dataset
from tfidf_embedding_vectorizer import TfidfEmbeddingVectorizer


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
        embedding = glove_dataset('../dataset/glove.840B.300d.txt')
    steps = [('vect', MeanEmbeddingVectorizer(embedding))]
    steps.append(('cls', classifier))
    return Pipeline(steps)


def glove_tfidf_vectorizer(classifier, word2vec=None):
    if word2vec:
        embedding = word2vec
    else:
        embedding = glove_dataset('../dataset/glove.840B.300d.txt')
    steps = [('vect', TfidfEmbeddingVectorizer(embedding))]
    steps.append(('cls', classifier))
    return Pipeline(steps)
