import os
import sys
import time
import pickle
import argparse
import numpy as np

from definitions import ROOT_PATH
from definitions import GLOVE_PATH
from definitions import WORD2VEC_PATH
from definitions import TRAIN_PATH
from definitions import TEST_PATH

from data.loaders import load_and_clean_data
from data.loaders import load_glove_embedding_matrix
from data.loaders import load_word2vec_embedding_matrix

from word_based_cnn import WordBasedCNN

from sklearn.model_selection import train_test_split

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

ROWS = None  # Load all reviews (~3.6M)

MAX_NUM_WORDS = 30000
MAX_SEQUENCE_LENGTH = 400
EMBEDDING_DIM = 32
EPOCHS = 5
VAL_SIZE = 0.05

EMBEDDING_TYPES = [
    "keras",
    "glove",
    "word2vec"
]

tokenizer = None


def train_tokenizer(samples, max_words):
    print("-- Training tokenizer")
    global tokenizer
    tokenizer = Tokenizer(num_words=max_words)
    tokenizer.fit_on_texts(samples)


def load_tokenizer(tokenizer_path):
    print("-- Loading tokenizer")
    with open(tokenizer_path, "rb") as file:
        global tokenizer
        tokenizer = pickle.load(file)


def reviews_to_sequences(reviews):
    print("-- Mapping reviews to sequences\n")

    global tokenizer
    sequences = tokenizer.texts_to_sequences(reviews)

    return pad_sequences(sequences,
                         maxlen=MAX_SEQUENCE_LENGTH,
                         padding="post")


def embedding_matrix(embedding_type="keras",
                     dim=None,
                     num_words=None,
                     word_index=None):

    if embedding_type == "keras":
        return None

    if embedding_type == "glove":
        matrix = load_glove_embedding_matrix(GLOVE_PATH)
    elif embedding_type == "word2vec":
        matrix = load_word2vec_embedding_matrix(WORD2VEC_PATH)
    else:
        raise Exception(
            "Unsupported embedding type: {}".format(embedding_type))

    embedding = np.zeros((num_words, dim))

    for word, idx in word_index.items():
        if idx >= num_words:
            continue
        embedding_vector = matrix.get(word)
        if embedding_vector is not None:
            embedding[idx] = embedding_vector

    return embedding


def parse_arguments():
    parser = argparse.ArgumentParser(description="Train word-based CNN")
    group = parser.add_mutually_exclusive_group()

    group.add_argument("--train", action="store_true")
    group.add_argument("--eval", action="store_true")

    parser.add_argument("--save", action="store_true")
    parser.add_argument("--tokenizer_path", type=str, default="tokenizer.pkl")
    parser.add_argument("--weights_path", type=str)

    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--rows", type=int, default=ROWS)
    parser.add_argument("--words", type=int, default=MAX_NUM_WORDS)
    parser.add_argument("--sequence_length", type=int,
                        default=MAX_SEQUENCE_LENGTH)
    parser.add_argument("--embedding", type=str, default=EMBEDDING_TYPES[0])
    parser.add_argument("--embedding_dim", type=int, default=EMBEDDING_DIM)
    parser.add_argument("--trainable_embeddings", action="store_true")
    parser.add_argument("--val_size", type=float, default=VAL_SIZE)

    return parser.parse_args()


def train(args):
    print("\n-- Training model --\n")
    print("-- Loading train data --")
    timestamp = int(time.time())

    train_samples, train_labels = load_and_clean_data(path=TRAIN_PATH,
                                                      nrows=args.rows)
    print("\n-- Found {} training samples".format(len(train_samples)))

    # train_tokenizer(samples=train_samples, max_words=args.words)

    tokenizer_path = os.path.join(
        ROOT_PATH, "models/{}".format(args.tokenizer_path))

    load_tokenizer(tokenizer_path)

    # if args.save:
    #     print("-- Saving tokenizer")
    #     tokenizer_path = os.path.join(ROOT_PATH, "models/tokenizer.pkl")
    #     with open(tokenizer_path, "wb") as file:
    #         pickle.dump(tokenizer, file, protocol=pickle.HIGHEST_PROTOCOL)

    train_sequences = reviews_to_sequences(train_samples)

    X_train, X_val, y_train, y_val = train_test_split(
        train_sequences, train_labels, test_size=args.val_size
    )

    num_words = min(args.words, len(tokenizer.word_index))
    embedding = embedding_matrix(embedding_type=args.embedding,
                                 dim=args.embedding_dim,
                                 num_words=num_words,
                                 word_index=tokenizer.word_index)

    model = WordBasedCNN(max_words=args.words,
                         max_sequence_length=args.sequence_length,
                         embedding_dim=args.embedding_dim,
                         embedding_matrix=embedding,
                         trainable_embeddings=args.trainable_embeddings,
                         save_best=args.save,
                         weights_path=os.path.join(
                             ROOT_PATH, "models/{}".format(args.weights_path)))

    training = model.fit(X_train, y_train,
                         X_val, y_val,
                         epochs=args.epochs)

    if args.save:
        print("\n-- Saving training history")
        history_filepath = "train_history_{}.pkl".format(timestamp)
        with open(history_filepath, "wb") as file:
            pickle.dump(training.history, file,
                        protocol=pickle.HIGHEST_PROTOCOL)


def evaluate(args):
    print("\n-- Evaluating model --\n")
    print("-- Loading test data")

    test_samples, test_labels = load_and_clean_data(path=TEST_PATH)

    print("\n-- Found {} test samples\n".format(len(test_samples)))

    tokenizer_path = os.path.join(
        ROOT_PATH, "models/{}".format(args.tokenizer_path))

    if tokenizer_path is not None and os.path.isfile(tokenizer_path):
        load_tokenizer(tokenizer_path)
    else:
        print("\n-- [ERROR] Failed to load tokenizer\n")
        sys.exit(0)

    model = WordBasedCNN(max_words=args.words,
                         max_sequence_length=args.sequence_length,
                         embedding_dim=args.embedding_dim,
                         trainable_embeddings=args.trainable_embeddings,
                         save_best=args.save,
                         weights_path=os.path.join(
                             ROOT_PATH, "models/{}".format(args.weights_path)))

    test_sequences = reviews_to_sequences(test_samples)
    loss, accuracy = model.evaluate(test_sequences, test_labels)

    print("-- Test loss {:.2f}".format(loss))
    print("-- Test accuracy {:.4f}".format(accuracy))


if __name__ == "__main__":
    args = parse_arguments()

    if args.train:
        train(args)

    if args.eval:
        evaluate(args)
