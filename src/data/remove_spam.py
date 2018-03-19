#!/usr/bin/env python3

import sys
import multiprocessing

import pandas as pd
import dask.dataframe as dd

from langdetect import detect
from dask.multiprocessing import get
from timeit import default_timer as timer

CPU_CORES = multiprocessing.cpu_count()


def detect_language(review, lang="en"):
    try:
        return detect(review) == lang
    except Exception:
        return False


def remove_spam(filename):
    df = pd.read_csv(filename, header=None)
    reviews = df[2]

    dask_df = dd.from_pandas(reviews, npartitions=CPU_CORES)

    non_en_indices = dask_df.map_partitions(
        lambda partition: partition.apply(detect_language)
    ).compute(get=get)

    # Non-english reviews are spam
    non_spam = df[non_en_indices]
    non_spam[0] -= 1  # Make sentiment binary. 1 - positive and 0 - negative

    print("-- Saving processed file...\n")
    non_spam.to_csv(filename, header=False, index=False)


def main(argv):
    if len(argv) == 1:
        print("\nNo files specified.")
        return

    for filename in argv[1:]:
        print("\n-- Processing '{}'--\n".format(filename))
        start = timer()
        remove_spam(filename)
        print("-- Done in {:.2f}s".format(timer() - start))


if __name__ == "__main__":
    main(sys.argv)
