import numpy as np
import pandas as pd
import multiprocessing
import dask.dataframe as dd
import data.preprocessor as preprocessor

from dask.multiprocessing import get
from gensim.models import KeyedVectors


def load_and_clean_data(path, options=(), nrows=None):
    print("\n-- Loading dataset")
    data_frame = pd.read_csv(path, nrows=nrows, header=None)
    data_frame.fillna("", inplace=True)

    print("-- Cleaning dataset")
    samples, labels = clean_data(data_frame, options)

    return samples, labels


def clean_data(data_frame, options, parallel=True):
    preprocessor.configure(options)

    # data_frame[1] contains review title
    # data_frame[2] contains review body
    reviews = data_frame[1] + " " + data_frame[2]

    if parallel:
        cpu_cores = multiprocessing.cpu_count()
        dask_df = dd.from_pandas(reviews, npartitions=cpu_cores)

        def clean_review(review):
            return preprocessor.clean(review)

        processed_df = dask_df.map_partitions(
            lambda df: df.apply(clean_review)
        ).compute(get=get)
    else:
        processed_df = reviews.apply(
            lambda review: preprocessor.clean(review)
        )

    return processed_df.values, data_frame.iloc[:, 0].values


def load_glove_embedding_matrix(path):
    print("\n-- Loading GloVe embedding matrix")
    embedding_matrix = {}
    with open(path, "rb") as f:
        for line in f:
            if line != 0:
                values = line.split()
                word = values[0].decode("utf-8")
                coefs = np.asarray(values[1:], dtype="float32")
                embedding_matrix[word] = coefs
    return embedding_matrix


def load_word2vec_embedding_matrix(path):
    print("\n-- Loading word2vec embedding matrix")
    w2v = KeyedVectors.load_word2vec_format(path, binary=True)
    embedding_matrix = {
        word: vector for word, vector in zip(w2v.index2entity, w2v.syn0)
    }
    return embedding_matrix
