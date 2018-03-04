import os.path

from definitions import ROOT_PATH

from keras.layers import Activation
from keras.layers import Conv1D
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Embedding
from keras.layers import MaxPooling1D
from keras.layers import GlobalMaxPooling1D
from keras.layers import Flatten
from keras.models import Sequential
from keras.callbacks import ModelCheckpoint


class WordBasedCNN(object):
    def __init__(self,
                 max_words,
                 max_sequence,
                 embedding_dim,
                 save_best=False,
                 deep_model=False,
                 weights_path=None):

        self.max_words = max_words
        self.max_sequence = max_sequence
        self.embedding_dim = embedding_dim
        self.weights_path = weights_path
        self.save_best = save_best

        self.checkpoint_path = os.path.join(
            ROOT_PATH,
            "models/{}.hdf5".format("deep" if deep_model else "shallow")
        )

        self.model = self._compile_model(deep_model)

    def fit(self, X_train, y_train, X_val, y_val, batch_size=96, epochs=5):
        return self.model.fit(X_train, y_train,
                              batch_size=batch_size,
                              epochs=epochs,
                              validation_data=(X_val, y_val),
                              callbacks=self._callbacks())

    def evaluate(self, X_test, y_test):
        return self.model.evaluate(X_test, y_test, verbose=1)

    def _compile_model(self, deep_model):
        if deep_model:
            model = self._create_deep_model()
        else:
            model = self._create_shallow_model()

        if self.weights_path is not None and os.path.isfile(self.weights_path):
            print("\n-- Loading pretrained model --\n")
            model.load_weights(self.weights_path)

        model.compile(loss="binary_crossentropy",
                      optimizer="rmsprop",
                      metrics=["accuracy"])
        return model

    def _create_shallow_model(self):
        model = Sequential()

        model.add(Embedding(self.max_words,
                            self.embedding_dim,
                            input_length=self.max_sequence))

        model.add(Conv1D(filters=256, kernel_size=10, activation="relu"))
        model.add(GlobalMaxPooling1D())

        model.add(Dense(units=512, activation="relu"))
        model.add(Dropout(0.5))

        model.add(Dense(units=1))
        model.add(Activation("sigmoid"))

        return model

    def _create_deep_model(self):
        model = Sequential()

        model.add(Embedding(self.max_words,
                            self.embedding_dim,
                            input_length=self.max_sequence))

        model.add(Conv1D(filters=256, kernel_size=7, activation="relu"))
        model.add(MaxPooling1D(3))

        model.add(Conv1D(filters=256, kernel_size=7, activation="relu"))
        model.add(MaxPooling1D(3))

        model.add(Conv1D(filters=256, kernel_size=3, activation="relu"))
        model.add(Conv1D(filters=256, kernel_size=3, activation="relu"))
        model.add(Conv1D(filters=256, kernel_size=3, activation="relu"))

        model.add(Conv1D(filters=256, kernel_size=3, activation="relu"))
        model.add(MaxPooling1D(3))

        model.add(Flatten())

        model.add(Dense(units=1024, activation="relu"))
        model.add(Dropout(0.5))

        model.add(Dense(units=1024, activation="relu"))
        model.add(Dropout(0.5))

        model.add(Dense(units=1))
        model.add(Activation("sigmoid"))

        return model

    def _callbacks(self):
        callbacks = []
        if self.save_best:
            callbacks.append(ModelCheckpoint(filepath=self.checkpoint_path,
                                             verbose=1,
                                             save_best_only=True))
        return callbacks
