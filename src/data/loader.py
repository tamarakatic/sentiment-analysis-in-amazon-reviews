import multiprocessing
import pandas as pd
import dask.dataframe as dd

from dask.multiprocessing import get
import data.preprocessor as preprocessor


def load_data(path, rows=None, clean=True):
    print("\n=> Loading dataset...")

    return pd.read_csv(path, nrows=rows, header=None)


def clean_data(data_frame, options, parallel=True):
    print("=> Cleaning dataset...")

    preprocessor.configure(options)

    # data_frame[0] contains reviews data
    reviews = data_frame[2]

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
