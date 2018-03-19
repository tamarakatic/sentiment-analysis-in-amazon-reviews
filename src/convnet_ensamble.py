import os
import pickle
import numpy as np

from keras.preprocessing.sequence import pad_sequences

from word_based_cnn import WordBasedCNN
from definitions import ROOT_PATH
from definitions import TEST_PATH
from data.loaders import load_and_clean_data

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

trained_embeddings = WordBasedCNN(max_words=MAX_NUM_WORDS,
                                  max_sequence_length=MAX_SEQUENCE_LENGTH,
                                  embedding_dim=30,
                                  trainable_embeddings=False,
                                  kernel_size=10,
                                  weights_path=os.path.join(
                                      ROOT_PATH, "models/{}".format("cnn_weights.hdf5")))

trained_embeddings_adam = WordBasedCNN(max_words=MAX_NUM_WORDS,
                                       max_sequence_length=MAX_SEQUENCE_LENGTH,
                                       embedding_dim=32,
                                       kernel_size=5,
                                       trainable_embeddings=False,
                                       weights_path=os.path.join(
                                           ROOT_PATH, "models/{}".format("cnn_weights_adam.hdf5")))

word2vec_embeddings = WordBasedCNN(max_words=MAX_NUM_WORDS,
                                   max_sequence_length=MAX_SEQUENCE_LENGTH,
                                   embedding_dim=300,
                                   trainable_embeddings=False,
                                   kernel_size=10,
                                   weights_path=os.path.join(
                                       ROOT_PATH, "models/{}".format("cnn_weights_word2vec.hdf5")))

models = [
    trained_embeddings.model,
    # trained_embeddings_adam.model,
    # word2vec_embeddings.model
]

predictions = np.zeros((3, test_sequences.shape[0]))

for i, model in enumerate(models):
    predictions[i, :] = model.predict(test_sequences).reshape(-1)

mean_predictions = np.mean(predictions, axis=0)
print(mean_predictions.shape)
mean_predictions = [float(round(x)) for x in mean_predictions]
accuracy = np.mean(mean_predictions == test_labels)
print("\n-- Test accuracy: {:.4f}".format(accuracy))
