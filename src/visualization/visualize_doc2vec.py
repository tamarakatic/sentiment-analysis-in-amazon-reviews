import os
import pandas as pd

from tensorboard import visualize_embedding
from src.definitions import TRAIN_PATH
from src.definitions import ROOT_PATH
from src.vectorizers import Doc2VecVectorizer
from src.data.loaders import load_and_clean_data
from src.data.preprocessor import Options

from gensim.models import Doc2Vec

MODEL_PATH = os.path.join(ROOT_PATH, "models/doc2vec.model")
DOC_COUNT = 10000

options = (
    Options.EMAILS,
    Options.EMOTICONS,
    Options.REPEATING_VOWELS,
    Options.URLS,
)

samples, labels = load_and_clean_data(TRAIN_PATH, options, DOC_COUNT)
model = Doc2Vec.load(MODEL_PATH)

docvecs = Doc2VecVectorizer(model).transform(samples)

sentiment_label = [(idx, labels[idx]) for idx in range(DOC_COUNT)]
metadata = pd.DataFrame(data=sentiment_label, columns=["Index", "Sentiment"])

visualize_embedding(embedding_matrix=docvecs,
                    tensor_name="doc2vec",
                    metadata=metadata)
