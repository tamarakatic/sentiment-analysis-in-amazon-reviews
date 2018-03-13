RAW_DATA = data/raw
EXTERNAL_DATA = data/external
PROCESSED_DATA = data/processed
MODELS = models

AMAZON_REVIEWS = $(RAW_DATA)/amazon_reviews.tar.gz
GLOVE = $(EXTERNAL_DATA)/glove.840B.300d.zip
WORD2VEC = $(EXTERNAL_DATA)/word2vec.300.bin
DOC2VEC = $(MODELS)/doc2vec_dbow_300.tar.gz

TRAIN = $(PROCESSED_DATA)/train.csv
TEST = $(PROCESSED_DATA)/test.csv
TOKENIZER = $(MODELS)/tokenizer.pkl

GOOGLE_BUCKET = https://storage.googleapis.com/awesome-public-datasets

DATA = $(AMAZON_REVIEWS) $(GLOVE) $(WORD2VEC) $(DOC2VEC)

.PHONY: all data process clean install

all: install data process

install:
	pip3 install -r requirements.txt
	pip3 install -e .

data: $(DATA) $(TOKENIZER)

process: $(TRAIN) $(TEST) src/data/remove_spam.py
	src/data/remove_spam.py $(TRAIN) $(TEST)

clean:
	rm -rf $(RAW_DATA)/*
	rm -rf $(EXTERNAL_DATA)/*
	rm -rf $(PROCESSED_DATA)/*

$(TRAIN): $(AMAZON_REVIEWS)
	tar -C $(PROCESSED_DATA) -xzvf $< amazon_review_polarity_csv/train.csv \
		--strip-components=1 --touch

$(TEST): $(AMAZON_REVIEWS)
	tar -C $(PROCESSED_DATA) -xzvf $< amazon_review_polarity_csv/test.csv \
		--strip-components=1 --touch

$(TOKENIZER): $(MODELS)/tokenizer.pkl.gz
	gunzip -kvf $<

$(AMAZON_REVIEWS):
	wget -O $@ $(GOOGLE_BUCKET)/amazon_review_polarity_csv.tar.gz
	touch $@

$(GLOVE):
	wget -O $@ http://nlp.stanford.edu/data/glove.840B.300d.zip
	touch $@
	gunzip -lf $@

$(WORD2VEC):
	wget -O $@.gz $(GOOGLE_BUCKET)/word2vec.300.bin.gz
	gunzip $@.gz -f

$(DOC2VEC):
	wget -O $@ $(GOOGLE_BUCKET)/doc2vec_dbow_300.tar.gz
	tar -C $(MODELS) -xzvf $@
