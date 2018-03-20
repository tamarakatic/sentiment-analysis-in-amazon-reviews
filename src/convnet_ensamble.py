import os
import pickle
import numpy as np

from word_based_cnn import WordBasedCNN
from definitions import ROOT_PATH
from definitions import TEST_PATH
from data.loaders import load_and_clean_data

from keras.preprocessing.sequence import pad_sequences

MAX_NUM_WORDS = 30000
MAX_SEQUENCE_LENGTH = 400

test_samples, test_labels = load_and_clean_data(path=TEST_PATH)

tokenizer_path = os.path.join(
    ROOT_PATH, "models/{}".format("tokenizer.pkl"))

with open(tokenizer_path, "rb") as file:
    tokenizer = pickle.load(file)

sequences = tokenizer.texts_to_sequences(test_samples)

test_sequences = pad_sequences(sequences,
                               maxlen=MAX_SEQUENCE_LENGTH,
                               padding="post")

# Each tuple is in format (weights, embedding dimension, kernel size)
model_params = [
    ("convnet_rmsprop30.hdf5", 30, 10),
    ("convnet_adam32.hdf5", 32, 5),
    ("convnet_glove.hdf5", 300, 5),
    ("convnet_word2vec.hdf5", 300, 10)
]

models = [WordBasedCNN(max_words=MAX_NUM_WORDS,
                       max_sequence_length=MAX_SEQUENCE_LENGTH,
                       embedding_dim=embedding_dim,
                       kernel_size=kernel_size,
                       weights_path=os.path.join(
                           ROOT_PATH, "models/{}".format(weights))).model
          for weights, embedding_dim, kernel_size in model_params]

predictions = np.zeros((len(models), test_sequences.shape[0]))

for i, model in enumerate(models):
    predictions[i, :] = model.predict(test_sequences, verbose=1).reshape(-1)

mean_predictions = [float(round(x)) for x in np.mean(predictions, axis=0)]
accuracy = np.mean(mean_predictions == test_labels)
print("\n-- Test accuracy: {:.4f}".format(accuracy))
