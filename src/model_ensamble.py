import os
import pickle
import numpy as np

from deep_word_models import ConvNet
from deep_word_models import SimpleLSTM
from deep_word_models import ConvNetLSTM

from definitions import ROOT_PATH
from definitions import TEST_PATH
from data.loaders import load_and_clean_data

from keras.preprocessing.sequence import pad_sequences

MAX_NUM_WORDS = 30000
MAX_SEQUENCE_LENGTH = 256

test_samples, test_labels = load_and_clean_data(path=TEST_PATH)

tokenizer_path = os.path.join(
    ROOT_PATH, "models/{}".format("tokenizer.pkl"))

with open(tokenizer_path, "rb") as file:
    tokenizer = pickle.load(file)

sequences = tokenizer.texts_to_sequences(test_samples)

test_sequences = pad_sequences(sequences,
                               maxlen=MAX_SEQUENCE_LENGTH,
                               padding="post")

# Sequences must be pre-padded with zeros for LSTM model
test_sequences_lstm = pad_sequences(sequences,
                                    maxlen=MAX_SEQUENCE_LENGTH,
                                    padding="pre")

# Each tuple is in format (weights_path, embedding dimension, kernel size)
model_params = [
    ("convnet_keras", 32, 5),
    ("convnet_lstm", 32, 5),
    ("lstm", 32, None)
]

models = []
for model_name, embedding_dim, kernel_size in model_params:
    model_arguments = {
        "max_words": MAX_NUM_WORDS,
        "max_sequence_length": MAX_SEQUENCE_LENGTH,
        "embedding_dim": embedding_dim,
        "weights_path": os.path.join(ROOT_PATH,
                                     "models/{}.h5".format(model_name))
    }

    if model_name == "convnet_lstm":
        model = ConvNetLSTM(**model_arguments)
    elif model_name == "lstm":
        model = SimpleLSTM(**model_arguments)
    else:
        model = ConvNet(kernel_size=kernel_size, **model_arguments)

    models.append((model_name, model))

predictions = np.zeros((len(models), test_sequences.shape[0]))

for i, (model_name, model) in enumerate(models):
    print("Evaluating: {}".format(model_name))

    if "lstm" in model_name:
        predictions[i, :] = model.predict(
            test_sequences_lstm, verbose=1).reshape(-1)
    else:
        predictions[i, :] = model.predict(
            test_sequences, verbose=1).reshape(-1)

    accuracy = np.mean(np.round(predictions[i, :]) == test_labels)
    print("Accuracy: {:.4f}\n".format(accuracy))

mean_predictions = [float(round(x)) for x in np.mean(predictions, axis=0)]
ensamble_accuracy = np.mean(mean_predictions == test_labels)
print("\nEnsamble accuracy: {:.4f}".format(ensamble_accuracy))
