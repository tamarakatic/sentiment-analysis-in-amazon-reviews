import os
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from data.embedding_loader import loading_embedding_dataset
from data.loader import load_data, clean_data
from data.preprocessor import Options
import numpy as np
from keras.utils import to_categorical
from keras.layers import Embedding, Conv1D, MaxPooling1D, Input, GlobalMaxPooling1D
from keras.layers import Dense, Dropout
from keras.models import Model

current_filepath = os.path.dirname(os.path.abspath(__file__))
ROOT_PATH = os.path.abspath(os.path.join(current_filepath, os.pardir))
ROWS = 10000000
MAX_NUM_WORDS = 50000
MAX_SEQUENCE_LENGTH = 300
VAlIDATION_SPLIT = 0.2
EMBEDDING_DIM = 300

print('Preparing embedding layer...')
embedding_index = loading_embedding_dataset('../dataset/glove.840B.300d.txt')
print('Found %s word vectors.' % len(embedding_index))

data_path = os.path.join(ROOT_PATH, 'dataset/data_all.csv')
data = load_data(path=data_path, rows=ROWS)

options = Options.all()
options.remove(Options.STEMMER)
# options.remove(Options.LEMMATIZER)
options.remove(Options.SPELLING)
options.remove(Options.STOPWORDS)

texts, labels = clean_data(data, options)
labels_index = 2

tokenizer = Tokenizer(num_words=MAX_NUM_WORDS)
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)

word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))

data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)
labels = to_categorical(np.asarray(labels))
print('Shape of data tensor: ', data.shape)
print('Shape of label tensor: ', labels.shape)

# split data into a training set and a validation set
indices = np.arange(data.shape[0])
np.random.shuffle(indices)
data = data[indices]
labels = labels[indices]
num_validation_samples = int(VAlIDATION_SPLIT * data.shape[0])

x_train = data[:-num_validation_samples]
y_train = labels[:-num_validation_samples]
x_val = data[-num_validation_samples:]
y_val = labels[-num_validation_samples:]

# prepare embedding matrix
num_words = min(MAX_NUM_WORDS, len(word_index))
embedding_matrix = np.zeros((num_words, EMBEDDING_DIM))
for word, i in word_index.items():
    if i >= MAX_NUM_WORDS:
        continue
    embedding_vector = embedding_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector

embedding_layer = Embedding(num_words,
                            EMBEDDING_DIM,
                            weights=[embedding_matrix],
                            input_length=MAX_SEQUENCE_LENGTH,
                            trainable=False)

print('Training model...')
sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
embedded_sequences = embedding_layer(sequence_input)
x = Conv1D(256, 5, activation='relu')(embedded_sequences)
x = MaxPooling1D(5)(x)
x = Conv1D(256, 5, activation='relu')(x)
x = MaxPooling1D(5)(x)
x = Conv1D(256, 5, activation='relu')(x)
x = GlobalMaxPooling1D()(x)
x = Dense(1024, activation='relu')(x)
x = Dropout(0.5)(x)
preds = Dense(labels_index, activation='softmax')(x)

model = Model(sequence_input, preds)
model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['acc'])

model.fit(x_train, y_train,
          batch_size=128,
          epochs=5,
          validation_data=(x_val, y_val))

model.save_weights('weights.hdf5')
