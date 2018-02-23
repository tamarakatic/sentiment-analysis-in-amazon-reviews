import os
import pipeline

from data.preprocessor import Options
from data.loader import load_data, clean_data

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

current_filepath = os.path.dirname(os.path.abspath(__file__))
ROOT_PATH = os.path.abspath(os.path.join(current_filepath, os.pardir))

ROWS = 300000


if __name__ == "__main__":
    data_path = os.path.join(ROOT_PATH, 'dataset/400_data.csv')
    data = load_data(path=data_path, rows=ROWS)

    options = Options.all()
    options.remove(Options.STEMMER)
    options.remove(Options.LEMMATIZER)
    options.remove(Options.SPELLING)
    options.remove(Options.STOPWORDS)

    samples, labels = clean_data(data, options)

    X_train, X_test, y_train, y_test = train_test_split(
        samples, labels, test_size=0.2, random_state=42
    )

    # bow_log_regression = pipeline.bag_of_words(
    #     classifier=LogisticRegression(),
    #     tf_idf=True
    # )
    #
    # bow_log_regression.set_params(
    #     vect__ngram_range=(1, 5),
    #     vect__max_features=500000,
    #     cls__C=10.0
    # )

    glove_log_regression = pipeline.glove_vectorizer(
        classifier=LogisticRegression()
    )

    # glove_log_regression.set_params(
    #     cls__C=1.0
    # )

    print("=> Training model...")
    glove_log_regression.fit(X_train, y_train)

    print("=> Validating model...")
    predictions = glove_log_regression.predict(X_test)

    report = classification_report(
        y_test,
        predictions,
        target_names=['Negative', 'Positive'],
        digits=3
    )

    print("\n{}".format(report))
