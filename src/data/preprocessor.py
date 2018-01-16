import re
from nltk.corpus import stopwords

NEGATIVE_CONSTRUCTS = set([
    "ain't",
    "can't",
    "cannot"
    "don't"
    "isn't"
    "mustn't",
    "needn't",
    "neither",
    "never",
    "no",
    "nobody",
    "none",
    "nothing",
    "nowhere"
    "shan't",
    "shouldn't",
    "wasn't"
    "weren't",
    "won't"
])

URLS = r'(https?:\/\/(?:www\.|(?!www))[a-zA-Z0-9][a-zA-Z0-9-]+' \
    '[a-zA-Z0-9]\.[^\s]{2,}|www\.[a-zA-Z0-9][a-zA-Z0-9-]+' \
    '[a-zA-Z0-9]\.[^\s]{2,}|https?:\/\/(?:www\.|(?!www))[a-zA-Z0-9]\.' \
    '[^\s]{2,}|www\.[a-zA-Z0-9]\.[^\s]{2,})'


POSITIVE_EMOTICONS = set([
    ':-)', ':)', ':-]', ':-3', ':3', ':^)',
    '8-)', '8)', '=]', '=)', ':-D', ':D', ':-))'
])

NEGATIVE_EMOTICONS = set([
    ':-(', ':(', ':-c', ':c', ':<', ':[', ':''-(',
    ':-[', ':-||', '>:[', ':{', '>:(', ':-|', ':|',
    ':/', ':-/', ':\'', '>:/', ':S'
])


class Preprocessor(object):
    def __init__(self):
        pass

    def replace_negative_constructs(self, sentence):
        """Replaces negative english constructs with 'not' word."""
        words = []
        for word in sentence.lower().split():
            if word in NEGATIVE_CONSTRUCTS:
                words.append('not')
            else:
                words.append(word)
        return ' '.join(words)

    def remove_repeating_vowels(self, sentence):
        """Removes unnecessary repeating vowels. i.e. cooooool -> cool"""
        return re.sub(r'(.)\1+', r'\1\1', sentence)

    def replace_emoticons_with_tags(self, sentence):
        words = sentence.split()

        for i, word in enumerate(words):
            if word in POSITIVE_EMOTICONS:
                words[i] = 'positive'
            if word in NEGATIVE_EMOTICONS:
                words[i] = 'negative'
        return ' '.join(words)

    def remove_whitespace(self, sentence):
        return re.sub(r'\s+', ' ', sentence)

    def remove_punctuation(self, sentence):
        return re.sub(r'[^\w\s\']', '', sentence)

    def remove_urls(self, sentence):
        return re.sub(URLS, '', sentence)

    def remove_emails(self, sentence):
        return re.sub(r'\S*@\S*\s?', '', sentence)

    def remove_stopwords(self, sentence):
        stop = set(stopwords.words('english'))
        words = sentence.lower().split()
        return ' '.join([word for word in words if word not in stop])
