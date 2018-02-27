import os
import pipeline
import multiprocessing
import numpy as np

from data.preprocessor import Options
from data.loader import load_data, clean_data
from word_embedding.embedding_loader import loading_embedding_dataset

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.ensemble import ExtraTreesClassifier
from gensim.models.word2vec import Word2Vec
from gensim.models.keyedvectors import KeyedVectors

current_filepath = os.path.dirname(os.path.abspath(__file__))
ROOT_PATH = os.path.abspath(os.path.join(current_filepath, os.pardir))

ROWS = 300000
WORKERS = multiprocessing.cpu_count()

if __name__ == "__main__":
    data_path = os.path.join(ROOT_PATH, 'dataset/data_all.csv')
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

    bow_log_regression_tfidf = pipeline.bag_of_words(
        classifier=LogisticRegression(),
        tf_idf=True
    )

    bow_linear_svc_tfidf = pipeline.bag_of_words(
        classifier=LinearSVC(kernel='linear'),
        tf_idf=True
    )

    bow_linear_svc_tfidf.set_params(
        vect__ngram_range=(1, 5),
        vect__max_features=500000,
        cls__C=10.0
    )

    bow_multi_nb = pipeline.glove_mean_vectorizer(
        classifier=MultinomialNB()
    )

    bow_multi_nb.set_params(
        vect__ngram_range=(1, 5),
        vect__max_features=500000
    )

    bow_bernulli_nb_tfidf = pipeline.bag_of_words(
        classifier=BernoulliNB(),
        tf_idf=True
    )

    bow_bernulli_nb_tfidf.set_params(
        vect__ngram_range=(1, 5),
        vect__max_features=500000
    )
    w2v_model = Word2Vec(np.hstack((X_train, X_test)),
                         size=100,
                         window=5,
                         min_count=WORKERS)
    w2v = {w: vec for w, vec in zip(w2v_model.wv.index2word, w2v_model.wv.syn0)}

    glove_mean_w2v_logistic_reg = pipeline.glove_mean_vectorizer(
        classifier=LogisticRegression(),
        word2vec=loading_embedding_dataset(
            '../dataset/GoogleNews-vectors-negative300.txt'
        )
    )

    print("=> Training model...")
    glove_mean_w2v_logistic_reg.fit(X_train, y_train)

    print("=> Validating model...")
    predictions = glove_mean_w2v_logistic_reg.predict(X_test)

    report = classification_report(
        y_test,
        predictions,
        target_names=['Negative', 'Positive'],
        digits=3
    )

    print("\n{}".format(report))
