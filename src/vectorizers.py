import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.base import TransformerMixin

from collections import defaultdict


class Doc2VecVectorizer(TransformerMixin):
    def __init__(self, doc2vec_model, infer_steps=3):
        self.model = doc2vec_model
        self.vector_size = len(doc2vec_model.wv.vectors[0])

        self.infer_steps = infer_steps
        self.alpha = 0.025
        self.min_alpha = 0.001

    def fit(self, X, y):
        return self

    def transform(self, docs):
        return self._infer_vectors(docs)

    def _infer_vectors(self, corpus):
        vectors = np.zeros((len(corpus), self.vector_size))
        for idx, doc in enumerate(corpus):
            vectors[idx] = self.model.infer_vector(doc.split(),
                                                   steps=self.infer_steps,
                                                   alpha=self.alpha,
                                                   min_alpha=self.min_alpha)
        return vectors


class MeanEmbeddingVectorizer(TransformerMixin):
    def __init__(self, embedding_matrix):
        self.embedding_matrix = embedding_matrix
        if len(embedding_matrix) > 0:
            self.dim = len(embedding_matrix[next(iter(embedding_matrix))])
        else:
            self.dim = 0

    def fit(self, X, y):
        return self

    def transform(self, X):
        vectors = np.zeros((len(X), self.dim))
        for idx, words in enumerate(X):
            word_vectors = [
                self.embedding_matrix[word]
                for word in words.split() if word in self.embedding_matrix
            ]
            if len(word_vectors) > 0:
                vectors[idx] = np.mean(word_vectors, axis=0)
        return vectors


class TfidfMeanEmbeddingVectorizer(TransformerMixin):
    def __init__(self, embedding_matrix):
        self.embedding_matrix = embedding_matrix
        self.tfidf_weights = None
        if len(embedding_matrix) > 0:
            self.dim = len(embedding_matrix[next(iter(embedding_matrix))])
        else:
            self.dim = 0

    def fit(self, X, y):
        tfidf = TfidfVectorizer()
        tfidf.fit(X)
        max_idf = max(tfidf.idf_)

        self.tfidf_weights = defaultdict(
            lambda: max_idf,
            [(word, tfidf.idf_[i]) for word, i in tfidf.vocabulary_.items()]
        )
        return self

    def transform(self, X):
        vectors = np.zeros((len(X), self.dim))
        for idx, words in enumerate(X):
            word_vectors = [
                self.embedding_matrix[word] * self.tfidf_weights[word]
                for word in words.split() if word in self.embedding_matrix
            ]
            if len(word_vectors) > 0:
                vectors[idx] = np.mean(word_vectors, axis=0)
        return vectors
