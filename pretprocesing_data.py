import pandas as pd
import re
import string
from nltk.corpus import stopwords


def read_csv_file(file_name):
    return pd.read_csv(file_name, nrows=1000, header=None)


def remove_whitespace(col):
    return re.sub('\s+', ' ', col)


def leave_signle_quotes():
    return re.sub('\'', '', string.punctuation + string.digits)


def remove_urls_from_column(col, table):
    regex = r"(https?:\/\/(?:www\.|(?!www))[a-zA-Z0-9][a-zA-Z0-9-]+" \
        "[a-zA-Z0-9]\.[^\s]{2,}|www\.[a-zA-Z0-9][a-zA-Z0-9-]+" \
        "[a-zA-Z0-9]\.[^\s]{2,}|https?:\/\/(?:www\.|(?!www))[a-zA-Z0-9]\." \
        "[^\s]{2,}|www\.[a-zA-Z0-9]\.[^\s]{2,})"
    return re.sub(regex, "", col).translate(table)


def remove_emails_from_column(col):
    return re.sub('\S*@\S*\s?', '', col)


def remove_stopwords(sentence):
    stop = set(stopwords.words('english'))
    return ' '.join([word for word in sentence.lower().split() if word not in stop])


for row in read_csv_file('english_test.csv').itertuples():
    table = str.maketrans('', '', leave_signle_quotes())

    second_col_no_whitespace = remove_whitespace(row._2)
    third_col_no_whitespace = remove_whitespace(row._3)

    second_col_no_urls = remove_urls_from_column(second_col_no_whitespace, table)
    third_col_no_urls = remove_urls_from_column(third_col_no_whitespace, table)

    second_col_no_emails = remove_emails_from_column(second_col_no_urls)
    third_col_no_emails = remove_emails_from_column(third_col_no_urls)

    second_col = remove_stopwords(second_col_no_emails).strip()
    third_col = remove_stopwords(third_col_no_emails).strip()
    print(str(row._1) + ' ' + second_col + ' ' + third_col)
