import numpy as np

from sklearn.base import TransformerMixin


class Doc2VecVectorizer(TransformerMixin):
    def __init__(self, doc2vec_model):
        self.model = doc2vec_model
        self.vector_size = len(doc2vec_model.docvecs[0])

    def fit(self, X, y):
        return self

    def transform(self, docs):
        return self._infer_vectors(docs)

    def _infer_vectors(self, corpus, alpha=0.05, min_alpha=0.001, steps=3):
        vectors = np.zeros((len(corpus), self.vector_size))
        for idx, doc in enumerate(corpus):
            vectors[idx] = self.model.infer_vector(
                doc.split(), steps=steps, alpha=alpha
            )
        return vectors
