import os
import pandas as pd

import pipeline
import data.preprocessor as preprocessor

from data.preprocessor import Options

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

current_filepath = os.path.dirname(os.path.abspath(__file__))
ROOT_PATH = os.path.abspath(os.path.join(current_filepath, os.pardir))


def load_data(path, rows=None):
    print("\n=> Loading dataset...")

    return pd.read_csv(path, nrows=rows, header=None)


def clean_data(data_frame, options):
    print("=> Processing dataset...")

    preprocessor.configure(options)

    # data_frame[0] contains reviews data
    processed_df = data_frame[2].apply(
        lambda review: preprocessor.clean(review)
    )

    return processed_df.values, data_frame.iloc[:, 0].values


if __name__ == "__main__":
    data_path = os.path.join(ROOT_PATH, 'dataset/data.csv')
    data = load_data(path=data_path, rows=225000)

    options = Options.all()
    options.remove(Options.STEMMER)
    options.remove(Options.SPELLING)

    samples, labels = clean_data(data, options)

    X_train, X_test, y_train, y_test = train_test_split(
        samples, labels, test_size=0.33, random_state=42
    )

    bow_log_regression = pipeline.bag_of_words(
        classifier=LogisticRegression(),
        tf_idf=True
    )

    bow_log_regression.set_params(
        vect__ngram_range=(1, 5),
        vect__max_features=500000,
        cls__C=10.0
    )

    print("=> Training model...")
    bow_log_regression.fit(X_train, y_train)

    print("=> Validating model...")
    predictions = bow_log_regression.predict(X_test)

    report = classification_report(
        y_test,
        predictions,
        target_names=['Negative', 'Positive'],
        digits=3
    )

    print("\n{}".format(report))
