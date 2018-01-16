import re

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
    "won't",
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
