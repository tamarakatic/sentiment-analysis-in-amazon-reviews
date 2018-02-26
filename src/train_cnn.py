import pickle

from data.loader import load_and_clean_data
from word_based_cnn import WordBasedCNN

from sklearn.model_selection import train_test_split

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences


ROWS = 200000

MAX_NUM_WORDS = 30000
MAX_SEQUENCE_LENGTH = 400
EMBEDDING_DIM = 300

texts, labels = load_and_clean_data(path="../dataset/data_all.csv",
                                    nrows=ROWS)

tokenizer = Tokenizer(num_words=MAX_NUM_WORDS)
tokenizer.fit_on_texts(texts)

sequences = tokenizer.texts_to_sequences(texts)
data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH, padding="post")

X_train, X_test, y_train, y_test = train_test_split(
    data, labels, test_size=0.1, random_state=42
)

model = WordBasedCNN(max_words=MAX_NUM_WORDS,
                     max_sequence=MAX_SEQUENCE_LENGTH,
                     embedding_dim=EMBEDDING_DIM,
                     deep_model=False)

history = model.fit(X_train, y_train, X_test, y_test)

print("\n=> Saving training history...")

with open("history.pkl", "wb") as file:
    pickle.dump(history.history, file)
