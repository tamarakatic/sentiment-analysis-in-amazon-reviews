from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from data.preprocessor import Preprocessor
from nltk.stem.snowball import SnowballStemmer
from nltk.stem import WordNetLemmatizer

from pprint import pprint
from time import time

import pandas as pd
import enchant

dictionary = enchant.Dict('en_US')
p = Preprocessor()
stemmer = SnowballStemmer('english')
lemmatizer = WordNetLemmatizer()

df = pd.read_csv('../dataset/small_train.csv', nrows=30000, header=None)


def preprocess(review):
    review = review.lower()

    no_repeating = p.remove_repeating_vowels(review)
    negative = p.replace_negative_constructs(no_repeating)
    no_emoticons = p.replace_emoticons_with_tags(negative)
    no_urls = p.remove_urls(no_emoticons)
    no_emails = p.remove_emails(no_urls)
    no_punc = p.remove_punctuation(no_emails)

    enchanted = []
    for word in no_punc.split():
        if dictionary.check(word):
            enchanted.append(word)

    no_stop = p.remove_stopwords(' '.join(enchanted))
    no_whitespace = p.remove_whitespace(no_stop)

    stemmed = [lemmatizer.lemmatize(word)
               for word in no_whitespace.split()]
    return ' '.join(stemmed)


processed_df = df[2].apply(lambda x: preprocess(x))

data = processed_df.values
labels = df.iloc[:, 0].values

pipeline = Pipeline([
    ('vect', CountVectorizer(ngram_range=(1, 3))),
    ('tfidf', TfidfTransformer()),
    ('clf',  MultinomialNB()),
])

parameters = {
    'vect__max_df': (0.75, 1.0),
    'vect__max_features': (50000, 80000, 100000),
    # 'vect__ngram_range': ((1, 3), (1, 2)),  # unigrams or bigrams
    # 'tfidf__use_idf': (True, False),
    # 'tfidf__norm': ('l1', 'l2'),
    # 'clf__alpha': (0.00001, 0.000001),
    # 'clf__max_iter': (50, 75, 100),
    'clf__alpha': (0.75, 1.0)
}

if __name__ == "__main__":
    # multiprocessing requires the fork to happen in a __main__ protected
    # block

    # find the best parameters for both the feature extraction and the
    # classifier
    grid_search = GridSearchCV(pipeline, parameters, n_jobs=-1, verbose=0)

    print("Performing grid search...")
    print("pipeline:", [name for name, _ in pipeline.steps])
    print("parameters:")
    pprint(parameters)
    t0 = time()
    grid_search.fit(data, labels)
    print("done in %0.3fs" % (time() - t0))
    print()

    print("Best score: %0.3f" % grid_search.best_score_)
    print("Best parameters set:")
    best_parameters = grid_search.best_estimator_.get_params()
    for param_name in sorted(parameters.keys()):
        print("\t%s: %r" % (param_name, best_parameters[param_name]))
