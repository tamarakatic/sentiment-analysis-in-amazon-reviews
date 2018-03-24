# Sentiment Analysis in Amazon Reviews

Solving classification problem for sentiment polarity of Amazon product reviews.

## Getting Started

These instructions will help you to set up your environment and run examples on your local machine.

### Prerequisites

* git
* make
* python 3.5+
* pip 3
* virtualenv

### Installing

Clone repo:

```bash
git clone https://github.com/tamarakatic/sentiment-analysis-in-amazon-reviews.git
cd sentiment-analysis-in-amazon-reviews
```

Create virtual environment:

```bash
virtualenv -p python3 env
source env/bin/activate
```

Install Dependencies:

```bash
make install
```

Download dataset and pretrained models:

```bash
make data
```

Process dataset:

```bash
make process
```

Or just run `make` in project root directory to run all above steps.

#### Installing Tensorflow

If you have CPU, install tensorflow for CPU:

```bash
pip3 install tensorflow
```

If you have NVIDIA CUDA enabled GPU, install tensorflow for GPU:

```bash
pip3 install tensorflow-gpu
```

Make sure you have required CUDA and cuDNN binaries for installed `tensorflow-gpu`.

#### Optional

Install `OpenBLAS` and `LAPACK` libraries for additional performance gain.

## Running models

You can run traditional models (e.g. n-grams + Logistic Regression) or pretrained Deep Learning models (e.g. 1D ConvNet).

### Traditional models

Run bag of words models (Logistic Regression, Linear SVM and Multinomial Naive Bayes):

```bash
python3 src/train_models.py --model=bow
```

Run ngrams + Gradient Boosting Classifier (because it takes a lot of time):

```bash
python3 src/train_models.py --model=gb
```

Run pretrained embeddings + Logistic Regression (Doc2Vec, Word2Vec and GloVe):

```bash
python3 src/train_models.py --model=embedding
```

These models are trained on 50000 reviews and for each model, classification report is printed (accuracy, f1-score and training time).

### Deep Learning models

Evaluate pretrained ConvNet models on test data.

Choose one of these examples:

1) To evaluate pretrained ConvNet with Keras embedding run:

```bash
python3 src/sentiment_cnn.py --eval --weights_path=convnet_adam32.hdf5 --embedding=keras --embedding_dim=32
```

2) To evaluate pretrained ConvNet with GloVe embedding run:

```bash
python3 src/sentiment_cnn.py --eval --weights_path=convnet_glove.hdf5 --embedding=glove --embedding_dim=300
```

3) To evaluate pretrained ConvNet with GloVe embedding run:

```bash
python3 src/sentiment_cnn.py --eval --weights_path=convnet_word2vec.hdf5 --embedding=word2vec --embedding_dim=300
```

#### ConvNet Ensamble

To form ensamble of ConvNets and get 1-2% more accuracy on test set run:

```bash
python3 src/convnet_ensamble.py
```
