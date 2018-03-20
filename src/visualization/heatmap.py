import os
import pickle
import numpy as np
import keras.backend as K
import matplotlib.pyplot as plt

from src.definitions import ROOT_PATH
from src.word_based_cnn import WordBasedCNN

from wordcloud import WordCloud
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import text_to_word_sequence

MAX_NUM_WORDS = 30000
MAX_SEQUENCE_LENGTH = 400
EMBEDDING_DIM = 32

REVIEW = ("Simple, Durable, Fun game for all ages."
          "This is an AWESOME game!"
          "Almost everyone know tic-tac-toe so "
          "it is EASY to learn and quick to play.")

tokenizer_path = os.path.join(ROOT_PATH, "models/tokenizer.pkl")
weights_path = os.path.join(ROOT_PATH, "models/convnet_adam32.hdf5")

with open(tokenizer_path, "rb") as file:
    TOKENIZER = pickle.load(file)

MODEL = WordBasedCNN(max_words=MAX_NUM_WORDS,
                     max_sequence_length=MAX_SEQUENCE_LENGTH,
                     embedding_dim=EMBEDDING_DIM,
                     trainable_embeddings=False,
                     kernel_size=5,
                     weights_path=weights_path).model


def generate_heatmap(review, conv_layer_name="conv1d_1"):
    sequences = TOKENIZER.texts_to_sequences([review])
    x = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH, padding="post")

    output_layer = MODEL.output
    conv_layer = MODEL.get_layer(conv_layer_name)

    grads = K.gradients(output_layer, conv_layer.output)[0]
    pooled_grads = K.mean(grads, axis=(0, 1, 2))

    iterate = K.function([MODEL.input], [pooled_grads,
                                         conv_layer.output[0]])

    pooled_grads_value, conv_layer_output_value = iterate([x])
    num_filters = conv_layer.output.shape[2]

    for i in range(num_filters):
        conv_layer_output_value[:, i] *= pooled_grads_value

    heatmap = np.mean(conv_layer_output_value, axis=-1)
    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap)

    num_words = len(sequences[0])
    return heatmap[:num_words]


def generate_wordcloud(words, heatmap):
    frequencies = dict(zip(words, heatmap * 100))

    wordcloud = WordCloud(background_color="white", max_words=len(words))
    wordcloud.fit_words(frequencies)
    return wordcloud


if __name__ == "__main__":
    review = ("Simple, Durable, Fun game for all ages."
              "This is an AWESOME game!"
              "Almost everyone know tic-tac-toe so "
              "it is EASY to learn and quick to play.")

    heatmap = generate_heatmap(review)

    words = text_to_word_sequence(review)
    wordcloud = generate_wordcloud(words, heatmap)

    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    plt.show()

    image = np.ones((1, 10)) * heatmap.reshape(-1, 1)
    plt.imshow(image)
    plt.xticks([])
    plt.yticks(range(len(words)), words)
    plt.show()
