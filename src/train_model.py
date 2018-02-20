import os
import pandas as pd
import data.preprocessor as preprocessor

from data.preprocessor import Options
from time import time

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC

from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline

current_filepath = os.path.dirname(os.path.abspath(__file__))
ROOT_PATH = os.path.abspath(os.path.join(current_filepath, os.pardir))


def load_data(path, rows=None):
    print("\n=> Loading dataset...")

    return pd.read_csv(path, nrows=rows, header=None)


def clean_data(data_frame, options):
    print("=> Processing dataset...")

    preprocessor.configure(options)
    processed_df = data_frame[2].apply(
        lambda review: preprocessor.clean(review)
    )

    return processed_df.values, data_frame.iloc[:, 0].values


def create_bow_pipeline(classifier):
    return Pipeline([
        ('vect', CountVectorizer()),
        ('tfidf', TfidfTransformer()),
        ('clf',  classifier),
    ])


if __name__ == "__main__":
    data_path = os.path.join(ROOT_PATH, 'dataset/data.csv')
    data = load_data(path=data_path, rows=20000)

    options = Options.all()
    options.remove(Options.STEMMER)
    options.remove(Options.SPELLING)

    samples, labels = clean_data(data, options)

    X_train, X_test, y_train, y_test = train_test_split(
        samples, labels, test_size=0.33, random_state=42
    )

    pipeline = create_bow_pipeline(classifier=MultinomialNB())
    parameters = {
        'vect__max_df': (0.75, 1.0),
        # 'vect__max_features': (50000, 80000, 100000),
        'vect__ngram_range': ((1, 3), (1, 2)),
        # 'clf__max_iter': (75, 100, 125),
        # 'clf__loss': ('hinge', 'squared_hinge'),
        'clf__alpha': (1.0, 2.0, 10.0)
        # 'clf__C': (1.0, 0.1, 10)
    }

    # Find the best parameters for both the feature extraction and the
    # classifier
    grid_search = GridSearchCV(
        pipeline, parameters, n_jobs=-1, verbose=1, cv=3
    )

    print("=> Performing grid search...\n")

    t = time()
    grid_search.fit(X_train, y_train)

    print("\n=> Done in {:.3f}s\n".format(time() - t))
    print("Best score: {:.3f}".format(grid_search.best_score_))
    print("Best parameters set:")

    best_parameters = grid_search.best_estimator_.get_params()

    for param_name in sorted(parameters.keys()):
        print("\t%s: %r" % (param_name, best_parameters[param_name]))

    predictions = grid_search.best_estimator_.predict(X_test)

    report = classification_report(
        y_test, predictions, target_names=['Negative', 'Positive']
    )

    print("\n{}".format(report))
