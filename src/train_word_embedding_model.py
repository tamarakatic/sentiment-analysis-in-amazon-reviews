import numpy as np

from definitions import GLOVE_PATH
from definitions import TRAIN_PATH

from data.loader import load_and_clean_data
from data.loader import load_glove_embedding_matrix

from word_embedding_based_cnn import WordEmbeddingBasedCNN

from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

ROWS = 50000
MAX_NUM_WORDS = 30000
MAX_SEQUENCE_LENGTH = 400
VAL_SPLIT = 0.05
EMBEDDING_DIM = 300


def load_glove_dataset():
    embedding_index = load_glove_embedding_matrix(GLOVE_PATH)
    print("Found {} word vectors.".format(len(embedding_index)))
    return embedding_index


def find_tokens(texts):
    tokenizer = Tokenizer(num_words=MAX_NUM_WORDS)
    tokenizer.fit_on_texts(texts)
    return tokenizer


def prepare_embedding_matrix(num_words, word_index):
    embedding_matrix = np.zeros((num_words, EMBEDDING_DIM))
    embedding_index = load_glove_dataset()

    for word, i in word_index.items():
        if i >= MAX_NUM_WORDS:
            continue
        embedding_vector = embedding_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector

    return embedding_matrix


if __name__ == "__main__":
    texts, labels = load_and_clean_data(TRAIN_PATH, nrows=ROWS)
    tokenizer = find_tokens(texts)

    sequences = tokenizer.texts_to_sequences(texts)
    word_index = tokenizer.word_index
    print("Found {} unique tokens.".format(len(word_index)))

    data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH, padding="post")
    print("Shape of data tensor: ", data.shape)

    num_words = min(MAX_NUM_WORDS, len(word_index))
    embedding_matrix = prepare_embedding_matrix(num_words, word_index)

    model = WordEmbeddingBasedCNN(num_words=num_words,
                                  embedding_dim=EMBEDDING_DIM,
                                  embedding_matrix=embedding_matrix,
                                  max_seq_length=MAX_SEQUENCE_LENGTH)

    x_train, x_val, y_train, y_val = train_test_split(data, labels,
                                                      test_size=VAL_SPLIT)

    print("Training model...")
    train_model = model.train_convnet_model(x_train, y_train, x_val, y_val)

    print("Save model...")
    train_model.model.save_weights("weights.hdf5")
