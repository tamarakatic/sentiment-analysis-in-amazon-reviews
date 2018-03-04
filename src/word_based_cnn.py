import os.path

from definitions import ROOT_PATH

from keras.layers import Activation
from keras.layers import Conv1D
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Embedding
from keras.layers import GlobalMaxPooling1D
from keras.models import Sequential
from keras.callbacks import ModelCheckpoint


class WordBasedCNN(object):
    def __init__(self,
                 max_words,
                 max_sequence_length,
                 embedding_dim,
                 embedding_matrix=None,
                 trainable_embeddings=True,
                 save_best=False,
                 weights_path=None):

        self.max_words = max_words
        self.max_sequence_length = max_sequence_length
        self.embedding_dim = embedding_dim
        self.embedding_matrix = embedding_matrix
        self.trainable_embeddings = trainable_embeddings
        self.weights_path = weights_path
        self.save_best = save_best

        self.checkpoint_path = os.path.join(
            ROOT_PATH,
            "models/{}.hdf5".format("cnn_weights")
        )

        self.model = self._compile_model()

    def fit(self, X_train, y_train, X_val, y_val, batch_size=96, epochs=5):
        return self.model.fit(X_train, y_train,
                              batch_size=batch_size,
                              epochs=epochs,
                              validation_data=(X_val, y_val),
                              callbacks=self._callbacks())

    def evaluate(self, X_test, y_test):
        return self.model.evaluate(X_test, y_test, verbose=1)

    def _compile_model(self):
        model = self._create_model()

        if self.weights_path is not None and os.path.isfile(self.weights_path):
            print("\n-- Loading pretrained model --\n")
            model.load_weights(self.weights_path)

        model.compile(loss="binary_crossentropy",
                      optimizer="rmsprop",
                      metrics=["accuracy"])
        return model

    def _create_model(self):
        model = Sequential()

        model.add(self._embedding_layer())

        model.add(Conv1D(filters=256, kernel_size=10, activation="relu"))
        model.add(GlobalMaxPooling1D())

        model.add(Dense(units=512, activation="relu"))
        model.add(Dropout(0.5))

        model.add(Dense(units=1))
        model.add(Activation("sigmoid"))

        return model

    def _embedding_layer(self):
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
        return embedding

    def _callbacks(self):
        callbacks = []
        if self.save_best:
            callbacks.append(ModelCheckpoint(filepath=self.checkpoint_path,
                                             verbose=1,
                                             save_best_only=True))
        return callbacks
