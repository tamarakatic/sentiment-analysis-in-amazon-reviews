RAW_DATA = data/raw
EXTERNAL_DATA = data/external
PROCESSED_DATA = data/processed

AMAZON_REVIEWS = $(RAW_DATA)/amazon_reviews.tar.gz
GLOVE = $(EXTERNAL_DATA)/glove.840B.300d.zip

TRAIN = $(PROCESSED_DATA)/train.csv
TEST = $(PROCESSED_DATA)/test.csv

DATA = $(AMAZON_REVIEWS) $(GLOVE)

.PHONY: all data process clean

all: data process

data: $(DATA)

process: $(TRAIN) $(TEST) src/data/remove_spam.py
	src/data/remove_spam.py $(TRAIN) $(TEST)

clean:
	rm -rf $(RAW_DATA)/*
	rm -rf $(EXTERNAL_DATA)/*
	rm -rf $(PROCESSED_DATA)/*

$(TRAIN): $(AMAZON_REVIEWS)
	tar -C $(PROCESSED_DATA) -xzf $< amazon_review_polarity_csv/train.csv \
		--strip-components=1 --touch

$(TEST): $(AMAZON_REVIEWS)
	tar -C $(PROCESSED_DATA) -xzf $< amazon_review_polarity_csv/test.csv \
		--strip-components=1 --touch

$(AMAZON_REVIEWS):
	wget -O $@ https://storage.googleapis.com/awesome-public-datasets/amazon_review_polarity_csv.tar.gz
	touch $@

$(GLOVE):
	wget -O $@ http://nlp.stanford.edu/data/glove.840B.300d.zip
	touch $@
	gunzip -lf $@
