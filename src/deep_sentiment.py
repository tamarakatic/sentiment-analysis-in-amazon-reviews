import os
import sys
import time
import pickle
import argparse
import numpy as np

from train_model import color_text

from definitions import ROOT_PATH
from definitions import GLOVE_PATH
from definitions import WORD2VEC_PATH
from definitions import TRAIN_PATH
from definitions import TEST_PATH

from data.loaders import load_and_clean_data
from data.loaders import load_glove_embedding_matrix
from data.loaders import load_word2vec_embedding_matrix

from deep_word_models import ConvNet
from deep_word_models import SimpleLSTM
from deep_word_models import ConvNetLSTM

from colorama import init as init_colorama
from colorama import Fore

from sklearn import metrics

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

init_colorama()

ROWS = None  # Load all reviews (~3.6M)
EPOCHS = 5
VAL_SPLIT = 0.05
BATCH_SIZE = 128
EMBEDDING_DIM = 32
MAX_NUM_WORDS = 30000
MAX_SEQUENCE_LENGTH = 256

EMBEDDING_TYPES = [
    "keras",
    "glove",
    "word2vec"
]


def train_tokenizer(samples, max_words, save=True):
    print("-- Training tokenizer")
    tokenizer = Tokenizer(num_words=max_words)
    tokenizer.fit_on_texts(samples)

    if save:
        print("-- Saving tokenizer")
        tokenizer_path = os.path.join(ROOT_PATH, "models/tokenizer.pkl")
        with open(tokenizer_path, "wb") as file:
            pickle.dump(tokenizer, file, protocol=pickle.HIGHEST_PROTOCOL)

    return tokenizer


def load_tokenizer(tokenizer_path):
    print("\n-- Loading tokenizer")
    with open(tokenizer_path, "rb") as file:
        tokenizer = pickle.load(file)
        return tokenizer


def reviews_to_sequences(reviews, tokenizer, padding="post"):
    print("-- Mapping reviews to sequences\n")
    sequences = tokenizer.texts_to_sequences(reviews)

    return pad_sequences(sequences,
                         maxlen=MAX_SEQUENCE_LENGTH,
                         padding=padding)


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
    parser = argparse.ArgumentParser()
    mode_group = parser.add_mutually_exclusive_group()

    mode_group.add_argument("--train", action="store_true")
    mode_group.add_argument("--test", action="store_true")

    model_group = parser.add_mutually_exclusive_group()
    model_group.add_argument("--convnet", action="store_true")
    model_group.add_argument("--lstm", action="store_true")
    model_group.add_argument("--convnet_lstm", action="store_true")

    parser.add_argument("--save", action="store_true")
    parser.add_argument("--weights_path", type=str)
    parser.add_argument("--checkpoint_path", type=str)
    parser.add_argument("--tokenizer_path", type=str, default="tokenizer.pkl")

    parser.add_argument("--rows", type=int, default=ROWS)
    parser.add_argument("--words", type=int, default=MAX_NUM_WORDS)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    parser.add_argument("--val_split", type=float, default=VAL_SPLIT)

    parser.add_argument("--embedding", type=str, default=EMBEDDING_TYPES[0])
    parser.add_argument("--embedding_dim", type=int, default=EMBEDDING_DIM)
    parser.add_argument("--trainable_embeddings", action="store_true")
    parser.add_argument("--sequence_length", type=int,
                        default=MAX_SEQUENCE_LENGTH)

    return parser.parse_args()


def train(tokenizer, args):
    print("-- Training model")
    timestamp = int(time.time())

    num_words = min(args.words, len(tokenizer.word_index))
    embedding = embedding_matrix(embedding_type=args.embedding,
                                 dim=args.embedding_dim,
                                 num_words=num_words,
                                 word_index=tokenizer.word_index)

    model_arguments = {
        "max_words": args.words,
        "max_sequence_length": args.sequence_length,
        "embedding_dim": args.embedding_dim,
        "embedding_matrix": embedding,
        "trainable_embeddings": args.trainable_embeddings,
        "save_best": args.save,
        "checkpoint_path": args.checkpoint_path
    }

    if args.lstm:
        model = SimpleLSTM(**model_arguments)
    elif args.convnet:
        model = ConvNet(**model_arguments)
    elif args.convnet_lstm:
        model = ConvNetLSTM(**model_arguments)
    else:
        print("\nModel must be specified [--convnet, --lstm, --convnet_lstm]")
        sys.exit(0)

    train_samples, train_labels = load_and_clean_data(path=TRAIN_PATH,
                                                      nrows=args.rows)
    print("-- Found {} training samples".format(len(train_samples)))

    padding = "post" if args.convnet else "pre"
    train_samples = reviews_to_sequences(train_samples, tokenizer, padding)

    training_data = model.fit(train_samples, train_labels,
                              batch_size=args.batch_size,
                              validation_split=args.val_split,
                              epochs=args.epochs)

    if args.save:
        print("\n-- Saving training history")
        history_filepath = "train_history_{}.pkl".format(timestamp)
        with open(history_filepath, "wb") as file:
            pickle.dump(training_data.history, file,
                        protocol=pickle.HIGHEST_PROTOCOL)


def test(tokenizer, args):
    print("-- Evaluating model")

    weights_path = os.path.join(
        ROOT_PATH, "models/{}".format(args.weights_path))

    model_arguments = {
        "max_words": args.words,
        "max_sequence_length": args.sequence_length,
        "embedding_dim": args.embedding_dim,
        "weights_path": weights_path
    }

    if args.lstm:
        model = SimpleLSTM(**model_arguments)
    elif args.convnet:
        model = ConvNet(**model_arguments)
    elif args.convnet_lstm:
        model = ConvNetLSTM(**model_arguments)
    else:
        print("\nModel must be specified [--convnet, --lstm, --convnet_lstm]")
        sys.exit(0)

    test_samples, test_labels = load_and_clean_data(path=TEST_PATH)
    print("-- Found {} test samples".format(len(test_samples)))

    padding = "post" if args.convnet else "pre"
    test_samples = reviews_to_sequences(test_samples, tokenizer, padding)

    predictions = model.predict(test_samples, verbose=1)
    predictions = [float(round(x[0])) for x in predictions]

    accuracy = metrics.accuracy_score(test_labels, predictions)
    precision = metrics.precision_score(test_labels, predictions)
    recall = metrics.recall_score(test_labels, predictions)
    f1_score = metrics.f1_score(test_labels, predictions)

    print(color_text("Accuracy:  ", color=Fore.GREEN) +
          color_text("{:.4f}".format(accuracy), color=Fore.RED))

    print(color_text("Precision: ", color=Fore.GREEN) +
          color_text("{:.4f}".format(precision), color=Fore.RED))

    print(color_text("Recall:    ", color=Fore.GREEN) +
          color_text("{:.4f}".format(recall), color=Fore.RED))

    print(color_text("F1-score:  ", color=Fore.GREEN) +
          color_text("{:.4f}".format(f1_score), color=Fore.RED))


if __name__ == "__main__":
    args = parse_arguments()

    tokenizer_path = os.path.join(
        ROOT_PATH, "models/{}".format(args.tokenizer_path))

    if tokenizer_path is not None and os.path.isfile(tokenizer_path):
        tokenizer = load_tokenizer(tokenizer_path)
    else:
        print("\n-- [ERROR] Failed to load tokenizer\n")
        sys.exit(0)

    if args.train:
        train(tokenizer, args)

    if args.test:
        test(tokenizer, args)
