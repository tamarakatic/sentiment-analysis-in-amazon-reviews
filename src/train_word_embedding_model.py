import os
import numpy as np

from sklearn.model_selection import train_test_split

from data.embedding_loader import loading_embedding_dataset
from data.loader import load_data
from data.loader import clean_data
from data.preprocessor import Options
from word_embedding_based_cnn import WordEmbeddingBasedCNN

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

current_filepath = os.path.dirname(os.path.abspath(__file__))
ROOT_PATH = os.path.abspath(os.path.join(current_filepath, os.pardir))

ROWS = 500000
MAX_NUM_WORDS = 30000
MAX_SEQUENCE_LENGTH = 400
VAlIDATION_SPLIT = 0.05
EMBEDDING_DIM = 300


def load_glove_dataset():
    embedding_index = loading_embedding_dataset(os.path.join(ROOT_PATH,
                                                'dataset/glove.840B.300d.txt'))
    print('Found {} word vectors.'.format(len(embedding_index)))
    return embedding_index


def load_and_clean_data():
    data_path = os.path.join(ROOT_PATH, 'dataset/data_all.csv')
    data = load_data(path=data_path, rows=ROWS)

    options = Options.all()
    options.remove(Options.STEMMER)
    options.remove(Options.LEMMATIZER)
    options.remove(Options.SPELLING)
    options.remove(Options.STOPWORDS)
    options.remove(Options.NEGATIVE_CONSTRUCTS)

    return clean_data(data, options)


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
    texts, labels = load_and_clean_data()
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
                                                      test_size=VAlIDATION_SPLIT)

    print("Training model...")
    train_model = model.train_convnet_model(x_train, y_train, x_val, y_val)

    print("Save model...")
    train_model.model.save_weights('weights.hdf5')
