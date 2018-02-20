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
