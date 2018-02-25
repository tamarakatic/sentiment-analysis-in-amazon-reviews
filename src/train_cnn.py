import pandas as pd

from sklearn.model_selection import train_test_split

from keras.layers import Activation
from keras.layers import Conv1D
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Embedding
from keras.layers import GlobalMaxPooling1D
from keras.models import Sequential
from keras.layers.normalization import BatchNormalization
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

from data.preprocessor import Options
from data.loader import clean_data

MAX_NUM_WORDS = 60000
MAX_SEQUENCE_LENGTH = 500
EMBEDDING_DIM = 50

df = pd.read_csv('../dataset/data_all.csv', nrows=800000, header=None)

options = set([
    Options.EMAILS,
    Options.EMOTICONS,
    Options.REPEATING_VOWELS,
    Options.URLS,
])

texts, labels = clean_data(df, options)

tokenizer = Tokenizer(num_words=MAX_NUM_WORDS)
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)

data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)

X_train, X_test, y_train, y_test = train_test_split(
    data, labels, test_size=0.1, random_state=42
)

model = Sequential()

model.add(Embedding(MAX_NUM_WORDS,
                    EMBEDDING_DIM,
                    input_length=MAX_SEQUENCE_LENGTH))

model.add(Dropout(0.4))

model.add(Conv1D(256,
                 10,
                 padding='valid',
                 strides=1))

model.add(BatchNormalization(axis=-1))
model.add(Activation('relu'))
model.add(GlobalMaxPooling1D())

model.add(Dense(256))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.5))

model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

model.fit(X_train, y_train,
          batch_size=64,
          epochs=5,
          validation_data=(X_test, y_test))

model.save_weights('../models/cnn_weights.h5')
