import pipeline
import multiprocessing

from definitions import TRAIN_PATH

from data.preprocessor import Options
from data.loader import load_and_clean_data
from data.loader import load_word2vec_embedding_matrix
from data.loader import load_glove_embedding_matrix

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.ensemble import ExtraTreesClassifier

from gensim.models.word2vec import Word2Vec
from gensim.models.keyedvectors import KeyedVectors

ROWS = 50000
WORKERS = multiprocessing.cpu_count()

if __name__ == "__main__":

    options = Options.all()
    options.remove(Options.STEMMER)
    options.remove(Options.SPELLING)
    options.remove(Options.STOPWORDS)

    samples, labels = load_and_clean_data(TRAIN_PATH, options, ROWS)

    X_train, X_test, y_train, y_test = train_test_split(
        samples, labels, test_size=0.2, random_state=42
    )

    bow_log_regression_tfidf = pipeline.bag_of_words(
        classifier=LogisticRegression(),
        tf_idf=True
    )

    bow_linear_svc_tfidf = pipeline.bag_of_words(
        classifier=LinearSVC(),
        tf_idf=True
    )

    bow_linear_svc_tfidf.set_params(
        vect__ngram_range=(1, 5),
        vect__max_features=500000,
        cls__C=10.0
    )

    bow_mnb_tfidf = pipeline.bag_of_words(
        classifier=MultinomialNB(),
        tf_idf=True
    )

    bow_mnb_tfidf.set_params(
        vect__ngram_range=(1, 5),
        vect__max_features=500000
    )

    glove = pipeline.glove_mean_embedding(
        classifier=LogisticRegression()
    )

    word2vec = pipeline.word2vec_mean_embedding(
        classifier=LogisticRegression(),
    )

    print("=> Training model...")
    glove.fit(X_train, y_train)

    print("=> Validating model...")
    predictions = glove.predict(X_test)

    report = classification_report(
        y_test,
        predictions,
        target_names=["Negative", "Positive"],
        digits=3
    )

    print("\n{}".format(report))
