import os
import multiprocessing
import numpy as np

from data.preprocessor import Options
from data.loader import load_data, clean_data

from texttable import Texttable

from gensim.test.test_doc2vec import ConcatenatedDoc2Vec
from gensim.models.doc2vec import TaggedDocument
from gensim.models import Doc2Vec

from sklearn import utils
from sklearn import linear_model
from sklearn.model_selection import train_test_split

from collections import OrderedDict
from collections import defaultdict

current_filepath = os.path.dirname(os.path.abspath(__file__))
ROOT_PATH = os.path.abspath(os.path.join(current_filepath, os.pardir))

WORKERS = multiprocessing.cpu_count()
ROWS = 30000
EPOCHS = 10

VECTOR_SIZES = [100, 200, 300, 400]
ALPHAS = [0.1, 0.08, 0.06, 0.04]


def labelize_reviews(reviews, label_type):
    labeled = []
    for idx, review in enumerate(reviews):
        label = "{}_{}".format(label_type, idx)
        labeled.append(TaggedDocument(review.split(), [label]))
    return labeled


def infer_vectors(model, corpus, vector_size, steps=3, alpha=0.05):
    vectors = np.zeros((len(corpus), vector_size))
    for idx, doc in enumerate(corpus):
        vectors[idx] = model.infer_vector(
            doc.words, steps=steps, alpha=alpha
        )
    return vectors


def validation_accuracy_for_model(model,
                                  X_train, y_train,
                                  X_val, y_val,
                                  classifier=linear_model.LogisticRegression()):

    vector_size = len(model.docvecs[0])
    train_vecs = infer_vectors(model, X_train, vector_size)
    val_vecs = infer_vectors(model, X_val, vector_size)

    classifier.fit(train_vecs, y_train)

    return classifier.score(val_vecs, y_val) * 100.0


def load_and_clean_data(path):
    data_path = os.path.join(ROOT_PATH, path)
    data = load_data(path=data_path, rows=ROWS)

    options = set([
        Options.EMOTICONS,
        Options.EMAILS,
        Options.URLS,
        Options.REPEATING_VOWELS,
    ])

    return clean_data(data, options)


def build_vocabulary(models, corpus):
    # Speed up setup by sharing results of the 1st model's vocabulary scan
    models[0].build_vocab(corpus)
    for model in models[1:]:
        model.reset_from(models[0])


def create_models(corpus, vector_size):
    models = [
        # PV-DM w/ concatenation
        Doc2Vec(dm=1, dm_concat=1, vector_size=vector_size, window=5,
                sample=1e-4, negative=5, hs=0, min_count=3, workers=WORKERS),

        # PV-DBOW
        Doc2Vec(dm=0, vector_size=vector_size, negative=5, hs=0,
                sample=1e-4, min_count=3, workers=WORKERS),

        # PV-DM w/ average
        Doc2Vec(dm=1, dm_mean=1, vector_size=vector_size, window=10,
                sample=1e-4, negative=5, hs=0, min_count=3, workers=WORKERS)
    ]

    build_vocabulary(models, corpus)

    models_by_name = OrderedDict((str(model), model) for model in models)
    models_by_name['dbow + dmm'] = ConcatenatedDoc2Vec([models[1], models[2]])
    models_by_name['dbow + dmc'] = ConcatenatedDoc2Vec([models[1], models[0]])
    return models_by_name


def cross_validate(samples, labels):
    X_train, X_test, y_train, y_test = train_test_split(
        samples, labels, test_size=0.1, random_state=42
    )

    train_corpus = labelize_reviews(X_train, 'TRAIN')
    test_corpus = labelize_reviews(X_test, 'TEST')
    corpus = labelize_reviews(samples, label_type='REVIEW')

    total_iterations = len(VECTOR_SIZES) * len(ALPHAS)
    iteration = 1

    print("=> Running cross validation...\n")

    best_results = defaultdict(lambda: (0.0, None, None))

    for vector_size in VECTOR_SIZES:
        for learning_rate in ALPHAS:
            print(("- Iteration {:2d}/{}: vector_size: {}" +
                   " - alpha: {}").format(iteration,
                                          total_iterations,
                                          vector_size,
                                          learning_rate))
            iteration += 1

            alpha, min_alpha, passes = (learning_rate, 0.001, EPOCHS)
            alpha_delta = (alpha - min_alpha) / passes

            models = create_models(corpus, vector_size)

            for epoch in range(passes):
                # Shuffling gets best results
                train_data = utils.shuffle(corpus)

                for name, model in models.items():
                    model.alpha, model.min_alpha = alpha, alpha

                    model.train(
                        train_data,
                        total_examples=len(train_data),
                        epochs=1
                    )
                alpha -= alpha_delta

            for name, model in models.items():
                acc = validation_accuracy_for_model(
                    model, train_corpus, y_train, test_corpus, y_test
                )

                if acc > best_results[name][0]:
                    best_results[name] = (acc, vector_size, learning_rate)
    return best_results


def print_results(results, save=False):
    table = Texttable()
    headings = ['Accuracy', 'Vector Size', 'Alpha', 'Model']
    table.header(headings)

    accuracies = []
    vector_sizes = []
    alphas = []
    models = []
    for accuracy, params, model_name in results:
        accuracies.append("{:.2f}%".format(accuracy))
        vector_sizes.append(params[0])
        alphas.append(params[1])
        models.append(model_name)

    for row in zip(accuracies, vector_sizes, alphas, models):
        table.add_row(row)

    print("\n{}".format(table.draw()))

    # Save results to csv file
    if save:
        import pandas as pd

        data = {
            'Accuracy': accuracies,
            'Vector Size': vector_sizes,
            'Alpha': alphas,
            'Model': models
        }

        df = pd.DataFrame(data, columns=headings)

        filename = 'doc2vec_cv_results.csv'
        df.to_csv(filename, index=False)

        print("\n=> Results saved to '{}'".format(filename))


if __name__ == '__main__':
    samples, labels = load_and_clean_data('dataset/data_all.csv')

    results = cross_validate(samples, labels)

    sorted_results = sorted(
        [(params[0], params[1:], name) for name, params in results.items()],
        reverse=True
    )

    print_results(sorted_results, save=True)
