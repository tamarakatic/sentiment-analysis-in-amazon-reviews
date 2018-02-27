import os
import multiprocessing

from timeit import default_timer as timer

from data.preprocessor import Options
from data.loader import load_and_clean_data

from gensim.models.doc2vec import TaggedDocument
from gensim.models import Doc2Vec

from sklearn import utils

current_filepath = os.path.dirname(os.path.abspath(__file__))
ROOT_PATH = os.path.abspath(os.path.join(current_filepath, os.pardir))

WORKERS = multiprocessing.cpu_count()
ROWS = 1000000

EPOCHS = 10
VECTOR_SIZE = 300
ALPHA = 0.1
MIN_COUNT = 3
SAMPLE = 1e-4


def label_reviews(reviews, label_type):
    labeled = []
    for idx, review in enumerate(reviews):
        label = "{}_{}".format(label_type, idx)
        labeled.append(TaggedDocument(review.split(), [label]))
    return labeled


def train_model(model, corpus, epochs,
                alpha=0.1, min_alpha=0.001,
                save=True, verbose=False):

    if verbose:
        print("\n- Starting training\n")

    start_time = timer()
    alpha_delta = (alpha - min_alpha) / epochs

    for epoch in range(epochs):
        start = timer()

        # Shuffling gets best results
        train_data = utils.shuffle(corpus)
        model.alpha, model.min_alpha = alpha, alpha

        model.train(
            train_data,
            total_examples=len(train_data),
            epochs=1
        )

        alpha -= alpha_delta
        end = timer()

        if verbose:
            print("- Epoch {:2d}/{} - elapsed: {:.2f}s".format(epoch + 1,
                                                               epochs,
                                                               end - start))
    training_time = timer() - start_time
    if verbose:
        print("\n- Training complete in {:.2f}s\n".format(training_time))

    if save:
        model_filename = "models/doc2vec.model"
        model_path = os.path.join(ROOT_PATH, model_filename)
        model.save(model_path)
        print("- Model saved to '{}'\n".format(model_filename))


def create_corpus():
    data_path = os.path.join(ROOT_PATH, "dataset/data_all.csv")

    options = set([
        Options.EMOTICONS,
        Options.EMAILS,
        Options.URLS,
        Options.REPEATING_VOWELS,
    ])

    samples, labels = load_and_clean_data(data_path, options, ROWS)
    corpus = label_reviews(samples, label_type="REVIEW")
    return corpus


def create_model(corpus):
    model = Doc2Vec(dm=0, negative=5, hs=0,
                    vector_size=VECTOR_SIZE,
                    sample=SAMPLE,
                    min_count=MIN_COUNT,
                    workers=WORKERS)

    model.build_vocab(corpus)
    return model


if __name__ == "__main__":
    corpus = create_corpus()
    model = create_model(corpus)

    print("\n-- {:7} {}".format("Model:", str(model)))
    print("-- {:7} {}".format("Epochs:", EPOCHS))
    print("-- {:7} {}".format("Alpha:", ALPHA))
    print("-- {:7} {}".format("Tokens:", len(model.wv.vocab)))

    train_model(model, corpus, epochs=EPOCHS, verbose=True)
