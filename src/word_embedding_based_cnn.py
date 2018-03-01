from keras.layers import Embedding
from keras.layers import Conv1D
from keras.layers import MaxPooling1D
from keras.layers import GlobalMaxPooling1D
from keras.layers import Dense
from keras.layers import Dropout
from keras.models import Sequential


class WordEmbeddingBasedCNN(object):
    def __init__(self,
                 num_words,
                 embedding_dim,
                 embedding_matrix,
                 max_seq_length):

        self.num_words = num_words
        self.embedding_dim = embedding_dim
        self.embedding_matrix = embedding_matrix
        self.max_seq_length = max_seq_length
        self.model = self._compile_convnet_model()

    def _compile_convnet_model(self):
        model = self._create_convnet_model()

        model.compile(loss='binary_crossentropy',
                      optimizer='rmsprop',
                      metrics=['accuracy'])

        return model

    def _create_convnet_model(self):
        model = Sequential()
        model.add(Embedding(self.num_words,
                            self.embedding_dim,
                            weights=[self.embedding_matrix],
                            input_length=self.max_seq_length,
                            trainable=True))

        model.add(Conv1D(filters=256, kernel_size=5, activation='relu'))
        model.add(MaxPooling1D(5))

        model.add(Conv1D(filters=256, kernel_size=5, activation='relu'))
        model.add(MaxPooling1D(5))

        model.add(Conv1D(filters=256, kernel_size=5, activation='relu'))
        model.add(GlobalMaxPooling1D())

        model.add(Dense(units=1024, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(units=1, activation='sigmoid'))

        return model

    def train_convnet_model(self, x_train, y_train, x_val, y_val):
        return self.model.fit(x_train,
                              y_train,
                              batch_size=96,
                              epochs=5,
                              validation_data=(x_val, y_val))
