import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import defaultdict


class MeanEmbeddingVectorizer(object):
    def __init__(self, word2vec):
        self.word2vec = word2vec
        if len(word2vec) > 0:
            self.dim = len(word2vec[next(iter(word2vec))])
        else:
            self.dim = 0

    def fit(self, X, y):
        return self

    def transform(self, X):
        return np.array([
            np.mean([self.word2vec[w] for w in words.split() if w in self.word2vec]
                    or [np.zeros(self.dim)], axis=0)
            for words in X
        ])


class TfidfEmbeddingVectorizer(object):
    def __init__(self, embedding):
        self.embedding = embedding
        self.tfidf_weights = None
        if len(embedding) > 0:
            self.dim = len(embedding[next(iter(embedding))])
        else:
            self.dim = 0

    def fit(self, X, y):
        tfidf = TfidfVectorizer()
        tfidf.fit(X)
        max_idf = max(tfidf.idf_)
        self.tfidf_weights = defaultdict(
            lambda: max_idf,
            [(w, tfidf.idf_[i]) for w, i in tfidf.vocabulary_.items()])
        return self

    def transform(self, X):
        return np.array([
                np.mean([self.embedding[w] * self.tfidf_weights[w]
                        for w in words.split() if w in self.embedding] or
                        [np.zeros(self.dim)], axis=0)
                for words in X
        ])
