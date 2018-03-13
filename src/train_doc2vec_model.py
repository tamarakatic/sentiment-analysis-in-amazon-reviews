import os
import multiprocessing

from timeit import default_timer as timer

from definitions import ROOT_PATH
from definitions import TRAIN_PATH

from data.preprocessor import Options
from data.loaders import load_and_clean_data

from gensim.models.doc2vec import TaggedDocument
from gensim.models import Doc2Vec

from sklearn import utils

WORKERS = multiprocessing.cpu_count()
ROWS = 1000000

EPOCHS = 250
ALPHA = 0.025
MIN_ALPHA = 0.0001

HIERARCHICAL_SOFTMAX = 0
MIN_COUNT = 3
NEGATIVE_SAMPLES = 5
SUBSAMPLE_THRESHOLD = 1e-4
VECTOR_SIZE = 300
WINDOW_SIZE = 10


def label_reviews(reviews, label_type):
    labeled = []
    for idx, review in enumerate(reviews):
        label = "{}_{}".format(label_type, idx)
        labeled.append(TaggedDocument(review.split(), [label]))
    return labeled


def train_model(model, corpus, epochs,
                alpha=ALPHA, min_alpha=MIN_ALPHA,
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
        model.delete_temporary_training_data(keep_doctags_vectors=False,
                                             keep_inference=True)
        model_filename = "models/doc2vec.model"
        model_path = os.path.join(ROOT_PATH, model_filename)
        model.save(model_path)
        print("- Model saved to '{}'\n".format(model_filename))


def create_corpus():
    options = (
        Options.EMAILS,
        Options.EMOTICONS,
        Options.REPEATING_VOWELS,
        Options.URLS,
    )

    samples, labels = load_and_clean_data(TRAIN_PATH, options, ROWS)
    corpus = label_reviews(samples, label_type="REVIEW")
    return corpus


def create_model(corpus):
    model = Doc2Vec(dm=0,
                    dbow_words=1,
                    hs=HIERARCHICAL_SOFTMAX,
                    min_count=MIN_COUNT,
                    negative=NEGATIVE_SAMPLES,
                    sample=SUBSAMPLE_THRESHOLD,
                    vector_size=VECTOR_SIZE,
                    window=WINDOW_SIZE,
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
