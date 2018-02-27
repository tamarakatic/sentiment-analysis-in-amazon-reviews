import re
import enchant

from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from nltk.stem import WordNetLemmatizer
from pipe import Pipe

from .patterns import NEGATIVE_CONSTRUCTS
from .patterns import NEGATIVE_EMOTICONS
from .patterns import POSITIVE_EMOTICONS
from .patterns import URLS


class Options(object):
    EMAILS = 'emails'
    EMOTICONS = 'emoticons'
    LEMMATIZER = 'lemmatizer'
    NEGATIVE_CONSTRUCTS = 'negative_constructs'
    PUNCTUATION = 'punctuation'
    REPEATING_VOWELS = 'repeating_vowels'
    SPELLING = 'spelling'
    STEMMER = 'stemmer'
    STOPWORDS = 'stopwords'
    URLS = 'urls'

    def all():
        return set([
            Options.EMAILS,
            Options.EMOTICONS,
            Options.LEMMATIZER,
            Options.NEGATIVE_CONSTRUCTS,
            Options.REPEATING_VOWELS,
            Options.SPELLING,
            Options.STEMMER,
            Options.STOPWORDS,
            Options.URLS
        ])


OPTIONS = Options.all()


def configure(options):
    global OPTIONS

    OPTIONS = options


def clean(sentence):
    return sentence.lower() \
        | remove_repeating_vowels \
        | replace_negative_constructs \
        | replace_emoticons_with_tags \
        | remove_urls \
        | remove_emails \
        | remove_punctuation \
        | remove_misspelled_words \
        | stem \
        | lemmatize


@Pipe
def remove_repeating_vowels(sentence):
    if Options.REPEATING_VOWELS not in OPTIONS:
        return sentence
    return re.sub(r'(.)\1+', r'\1\1', sentence)


@Pipe
def replace_negative_constructs(sentence):
    if Options.NEGATIVE_CONSTRUCTS not in OPTIONS:
        return sentence

    words = []
    for word in sentence.lower().split():
        if word in NEGATIVE_CONSTRUCTS:
            words.append('not')
        else:
            words.append(word)
    return ' '.join(words)


@Pipe
def replace_emoticons_with_tags(sentence):
    if Options.EMOTICONS not in OPTIONS:
        return sentence

    words = sentence.split()
    for i, word in enumerate(words):
        if word in POSITIVE_EMOTICONS:
            words[i] = 'positive'
        if word in NEGATIVE_EMOTICONS:
            words[i] = 'negative'
    return ' '.join(words)


@Pipe
def remove_urls(sentence):
    if Options.URLS not in OPTIONS:
        return sentence
    return re.sub(URLS, '', sentence)


@Pipe
def remove_emails(sentence):
    if Options.EMAILS not in OPTIONS:
        return sentence
    return re.sub(r'\S*@\S*\s?', '', sentence)


@Pipe
def remove_stopwords(sentence):
    if Options.STOPWORDS not in OPTIONS:
        return sentence

    stop = set(stopwords.words('english'))
    words = sentence.lower().split()
    return ' '.join([word for word in words if word not in stop])


LEMMATIZER = None


@Pipe
def lemmatize(sentence):
    if Options.LEMMATIZER not in OPTIONS:
        return sentence

    global LEMMATIZER
    if LEMMATIZER is None:
        LEMMATIZER = WordNetLemmatizer()

    lemmatized = [LEMMATIZER.lemmatize(word, pos='v')
                  for word in sentence.split()]
    return ' '.join(lemmatized)


STEMMER = None


@Pipe
def stem(sentence):
    if Options.STEMMER not in OPTIONS:
        return sentence

    global STEMMER
    if STEMMER is None:
        STEMMER = SnowballStemmer('english')

    stemmed = [STEMMER.stem(word) for word in sentence.split()]
    return ' '.join(stemmed)


DICTIONARY = None


@Pipe
def remove_misspelled_words(sentence):
    if Options.SPELLING not in OPTIONS:
        return sentence

    global DICTIONARY
    if DICTIONARY is None:
        DICTIONARY = enchant.Dict('en_US')

    return [word for word in sentence.split() if DICTIONARY.check(word)]


@Pipe
def remove_whitespace(sentence):
    return re.sub(r'\s+', ' ', sentence)


@Pipe
def remove_punctuation(sentence):
    if Options.PUNCTUATION not in OPTIONS:
        return sentence

    return re.sub(r'[^\w\s\']', '', sentence)
