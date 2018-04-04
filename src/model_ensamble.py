import os
import pickle
import numpy as np

from src.deep_word_models import ConvNet
from src.deep_word_models import SimpleLSTM
from src.deep_word_models import ConvNetLSTM

from src.definitions import ROOT_PATH
from src.definitions import TEST_PATH
from src.data.loaders import load_and_clean_data

from keras.preprocessing.sequence import pad_sequences

MAX_NUM_WORDS = 30000
MAX_SEQUENCE_LENGTH = 256

# Each tuple is in format (weights_path, embedding dimension, kernel size)
MODEL_PARAMS = [
    ("convnet_keras", 32, 5),
    ("convnet_lstm", 32, 5),
    ("lstm", 32, None)
]

# Model ensamble weights obtained with SLSQP optimizer
# See notebooks/6.0-model-ensamble-optimization.ipynb for details
WEIGHTS = [
    0.32983265,
    0.33383343,
    0.33633392
]


def init_models():
    models = []
    for model_name, embedding_dim, kernel_size in MODEL_PARAMS:
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

    return models


def models_prediction(sequences, labels, models):
    test_sequences = pad_sequences(sequences,
                                   maxlen=MAX_SEQUENCE_LENGTH,
                                   padding="post")
    # Sequences must be pre-padded with zeros for LSTM model
    test_sequences_lstm = pad_sequences(sequences,
                                        maxlen=MAX_SEQUENCE_LENGTH,
                                        padding="pre")

    predictions = np.zeros((len(models), test_sequences.shape[0]))

    for i, (model_name, model) in enumerate(models):
        print("Evaluating: {}".format(model_name))

        if "lstm" in model_name:
            predictions[i, :] = model.predict(
                test_sequences_lstm, verbose=1).reshape(-1)
        else:
            predictions[i, :] = model.predict(
                test_sequences, verbose=1).reshape(-1)

        accuracy = np.mean(np.round(predictions[i, :]) == labels)
        print("Accuracy: {:.4f}\n".format(accuracy))

    return predictions


def ensamble_prediction(model_predictions, weights=WEIGHTS):
    weights = np.array(weights).reshape(-1, 1)
    predictions = weights * model_predictions
    predictions = [float(round(x)) for x in np.sum(predictions, axis=0)]
    return predictions


def evaluate(samples, labels):
    models = init_models()

    tokenizer_path = os.path.join(
        ROOT_PATH, "models/{}".format("tokenizer.pkl"))

    with open(tokenizer_path, "rb") as file:
        tokenizer = pickle.load(file)

    sequences = tokenizer.texts_to_sequences(samples)

    predictions = models_prediction(sequences, test_labels, models)
    predictions = ensamble_prediction(predictions, test_labels)
    ensamble_accuracy = np.mean(predictions == labels)
    return ensamble_accuracy


if __name__ == "__main__":
    test_samples, test_labels = load_and_clean_data(path=TEST_PATH)
    ensamble_accuracy = evaluate(test_samples, test_labels)
    print("\nEnsamble accuracy: {:.4f}".format(ensamble_accuracy))
