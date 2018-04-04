import os.path

from src.definitions import ROOT_PATH

from keras.layers import Conv1D
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Embedding
from keras.layers import GlobalMaxPooling1D
from keras.layers import LSTM
from keras.layers import MaxPooling1D
from keras.models import Sequential
from keras.callbacks import ModelCheckpoint


class WordBasedModel(Sequential):

    def __init__(self,
                 max_words,
                 max_sequence_length,
                 embedding_dim,
                 checkpoint_path=None,
                 save_best=False,
                 embedding_matrix=None,
                 trainable_embeddings=False,
                 weights_path=None):

        super().__init__()

        self.max_words = max_words
        self.max_sequence_length = max_sequence_length
        self.embedding_dim = embedding_dim
        self.embedding_matrix = embedding_matrix
        self.trainable_embeddings = trainable_embeddings
        self.weights_path = weights_path
        self.save_best = save_best

        if checkpoint_path:
            self.checkpoint_path = os.path.join(
                ROOT_PATH,
                "models/{}".format(checkpoint_path)
            )

        self._compile_model()

    def fit(self, *args, **kwargs):
        return super().fit(callbacks=self._callbacks(), *args, **kwargs)

    def _compile_model(self):
        self.add_embedding_layer()
        self.add_model_layers()  # Subclasses must implement this method

        if self.weights_path is not None and os.path.isfile(self.weights_path):
            print("\n-- Loading pretrained model --\n")
            self.load_weights(self.weights_path)

        self.compile(loss="binary_crossentropy",
                     optimizer="adam",
                     metrics=["accuracy"])
        self.summary()

    def add_embedding_layer(self):
        if self.embedding_matrix is None:
            embedding = Embedding(input_dim=self.max_words,
                                  output_dim=self.embedding_dim,
                                  input_length=self.max_sequence_length)
        else:
            embedding = Embedding(input_dim=self.max_words,
                                  output_dim=self.embedding_dim,
                                  input_length=self.max_sequence_length,
                                  weights=[self.embedding_matrix],
                                  trainable=self.trainable_embeddings)

        self.add(embedding)

    def add_model_layers(self):
        raise NotImplementedError()

    def _callbacks(self):
        callbacks = []
        if self.save_best:
            callbacks.append(ModelCheckpoint(filepath=self.checkpoint_path,
                                             verbose=1,
                                             save_best_only=True))
        return callbacks


class ConvNetLSTM(WordBasedModel):
    FILTERS = 64
    KERNEL_SIZE = 5
    LSTM_UNITS = 128

    def add_model_layers(self):
        self.add(Conv1D(filters=self.FILTERS,
                        kernel_size=self.KERNEL_SIZE,
                        activation="relu"))
        self.add(MaxPooling1D(4))
        self.add(LSTM(units=self.LSTM_UNITS))
        self.add(Dense(units=1, activation="sigmoid"))


class ConvNet(WordBasedModel):
    FILTERS = 256

    def __init__(self, kernel_size=5, **kwargs):
        self.kernel_size = kernel_size
        super().__init__(**kwargs)

    def add_model_layers(self):
        self.add(Conv1D(filters=self.FILTERS,
                        kernel_size=self.kernel_size,
                        activation="relu"))

        self.add(GlobalMaxPooling1D())
        self.add(Dense(units=512, activation="relu"))
        self.add(Dropout(0.5))
        self.add(Dense(units=1, activation="sigmoid"))


class SimpleLSTM(WordBasedModel):
    LSTM_UNITS = 128

    def add_model_layers(self):
        self.add(LSTM(units=self.LSTM_UNITS))
        self.add(Dense(units=1, activation="sigmoid"))
