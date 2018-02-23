import os
import multiprocessing
import numpy as np

from data.preprocessor import Options
from data.loader import load_data, clean_data

from gensim.models.doc2vec import TaggedDocument
from gensim.models import Doc2Vec

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

from sklearn import utils

current_filepath = os.path.dirname(os.path.abspath(__file__))
ROOT_PATH = os.path.abspath(os.path.join(current_filepath, os.pardir))


WORKERS = multiprocessing.cpu_count()
ROWS = 150000
VECTOR_SIZE = 100


def labelize_reviews(reviews, label_type):
    labeled = []
    for idx, review in enumerate(reviews):
        label = "{}_{}".format(label_type, idx)
        labeled.append(TaggedDocument(review.split(), [label]))
    return labeled


def get_vectors(model, corpus, size, label_type):
    vectors = np.zeros((len(corpus), size))
    for idx in range(len(corpus)):
        label = "{}_{}".format(label_type, idx)
        vectors[idx] = model.docvecs[label]
    return vectors


if __name__ == '__main__':
    data_path = os.path.join(ROOT_PATH, 'dataset/200_data.csv')
    data = load_data(path=data_path, rows=ROWS)

    options = Options.all()
    options.remove(Options.STEMMER)
    options.remove(Options.LEMMATIZER)
    options.remove(Options.SPELLING)

    samples, labels = clean_data(data, options)

    X_train, X_test, y_train, y_test = train_test_split(
        samples, labels, test_size=0.2, random_state=42
    )

    train_corpus = labelize_reviews(X_train, 'TRAIN')
    test_corpus = labelize_reviews(X_test, 'TEST')
    corpus = train_corpus + test_corpus

    EPOCHS = 30

    model_ug_dbow = Doc2Vec(dm=0, hs=0, vector_size=VECTOR_SIZE, negative=5,
                            min_count=2, workers=WORKERS,
                            min_alpha=0.065, alpha=0.065)

    print("=> Building vocabulary...")
    model_ug_dbow.build_vocab(corpus)

    print("=> Training on TRAIN sentences...")
    for epoch in range(EPOCHS):
        model_ug_dbow.train(utils.shuffle(corpus),
                            total_examples=len(corpus), epochs=1)
        model_ug_dbow.alpha -= 0.002
        model_ug_dbow.min_alpha = model_ug_dbow.alpha

    train_vecs_dbow = get_vectors(model_ug_dbow, X_train, VECTOR_SIZE, "TRAIN")
    test_vecs_dbow = get_vectors(model_ug_dbow, X_test, VECTOR_SIZE, "TEST")

    classifier = LogisticRegression()

    print("=> Training classifier...")
    classifier.fit(train_vecs_dbow, y_train)

    predictions = classifier.predict(test_vecs_dbow)
    report = classification_report(
        y_test,
        predictions,
        target_names=['Negative', 'Positive'],
        digits=3
    )

    print("\n{}".format(report))
