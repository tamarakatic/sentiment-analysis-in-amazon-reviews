import os

from doc2vec_vectorizer import Doc2VecVectorizer
from gensim.models import Doc2Vec
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

current_filepath = os.path.dirname(os.path.abspath(__file__))
ROOT_PATH = os.path.abspath(os.path.join(current_filepath, os.pardir))


def bag_of_words(classifier, tf_idf=True):
    if tf_idf:
        steps = [('vect', TfidfVectorizer())]
    else:
        steps = [('vect', CountVectorizer())]

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
