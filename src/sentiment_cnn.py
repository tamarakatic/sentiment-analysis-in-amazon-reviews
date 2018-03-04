import os
import sys
import time
import pickle
import argparse

from definitions import ROOT_PATH
from definitions import TRAIN_PATH
from definitions import TEST_PATH
from data.loader import load_and_clean_data
from word_based_cnn import WordBasedCNN

from sklearn.model_selection import train_test_split

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

ROWS = None  # Pandas loads all reviews (~3.6M)

MAX_NUM_WORDS = 30000
MAX_SEQUENCE_LENGTH = 400
EMBEDDING_DIM = 30
EPOCHS = 5
VAL_SIZE = 0.05

tokenizer = Tokenizer(num_words=MAX_NUM_WORDS)


def parse_arguments():
    parser = argparse.ArgumentParser(description="Train word-based CNNs")
    group = parser.add_mutually_exclusive_group()

    group.add_argument("--train", action="store_true")
    group.add_argument("--eval", action="store_true")

    parser.add_argument("--save", action="store_true")
    parser.add_argument("--model_type", type=str, default="shallow")
    parser.add_argument("--tokenizer_path", type=str, default="tokenizer.pkl")
    parser.add_argument("--weights_path", type=str)

    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--rows", type=int, default=ROWS)
    parser.add_argument("--words", type=int, default=MAX_NUM_WORDS)
    parser.add_argument("--sequence_len", type=int,
                        default=MAX_SEQUENCE_LENGTH)
    parser.add_argument("--embedding_dim", type=int, default=EMBEDDING_DIM)
    parser.add_argument("--val_size", type=float, default=VAL_SIZE)

    return parser.parse_args()


def train_tokenizer(samples):
    print("-- Training tokenizer")
    global tokenizer
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


if __name__ == "__main__":
    timestamp = int(time.time())

    args = parse_arguments()

    model = WordBasedCNN(max_words=args.words,
                         max_sequence=args.sequence_len,
                         embedding_dim=args.embedding_dim,
                         save_best=args.save,
                         weights_path=os.path.join(
                             ROOT_PATH, "models/{}".format(args.weights_path)),
                         deep_model=False)

    if args.train:
        print("\n-- Training model\n")
        print("\n-- Loading train data --")

        train_samples, train_labels = load_and_clean_data(path=TRAIN_PATH,
                                                          nrows=args.rows)
        print("\n-- Found {} training samples".format(len(train_samples)))

        train_tokenizer(samples=train_samples)

        if args.save:
            print("-- Saving tokenizer")
            tokenizer_path = os.path.join(ROOT_PATH, "models/tokenizer.pkl")
            with open(tokenizer_path, "wb") as file:
                pickle.dump(tokenizer, file, protocol=pickle.HIGHEST_PROTOCOL)

        train_sequences = reviews_to_sequences(train_samples)

        X_train, X_val, y_train, y_val = train_test_split(train_sequences,
                                                          train_labels,
                                                          test_size=args.val_size)
        history = model.fit(X_train, y_train, X_val, y_val, epochs=args.epochs)

        if args.save:
            print("\n-- Saving training history")
            history_filepath = "train_history_{}.pkl".format(timestamp)
            with open(history_filepath, "wb") as file:
                pickle.dump(history.history, file,
                            protocol=pickle.HIGHEST_PROTOCOL)

    if args.eval:
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

        test_sequences = reviews_to_sequences(test_samples)
        loss, accuracy = model.evaluate(test_sequences, test_labels)

        print("-- Test loss {:.2f}".format(loss))
        print("-- Test accuracy {:.4f}".format(accuracy))
