import pandas as pd
from langdetect import detect
import sys

print('\n=> Loading dataset ...')

df = pd.read_csv('dataset/test.csv', header=None)
total_rows = df.shape[0]

current_row = 0

print('=> Dataset loaded!\n')


def detect_en_language(review):
    global current_row
    progress = round((100 * current_row) / total_rows, 2)
    sys.stdout.write("=> {}% \r".format(progress))
    sys.stdout.flush()
    current_row += 1

    try:
        return detect(review) == 'en'
    except ValueError:
        return False


print('=> Removing non-english reviews ...')

english_reviews = df[df[2].apply(detect_en_language)]

print('\n=> Writing to CSV file ...')

english_reviews.to_csv('english_test.csv', header=False, index=False, sep=',')

print('\n=> Done!')
