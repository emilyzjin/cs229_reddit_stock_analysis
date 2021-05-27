# Utility functions for data preprocessing for deep learning sentiment analysis model.

import string
from string import punctuation
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import numpy as np
import torch

lemmatizer = WordNetLemmatizer()


# def tokenize(s, stopwords=stopwords):
#     """
#     Given a string, remove stop words, lowercase words,
#     and remove words with non-alphanumeric characters.
#     """
#     tokenized = s.split()
#     result = []
#     for word in tokenized:
#         word = word.strip(punctuation)
#         if word.isalpha() and word not in stopwords:
#             result.append(lemmatizer.lemmatize(word).lower())
#     return result


def preprocess_string(s):
    # Remove all non-word characters (everything except numbers and letters)
    s = nltk.re.sub(r"[^\w\s]", '', s)
    # Replace all runs of whitespaces with no space
    s = nltk.re.sub(r"\s+", '', s)
    # replace digits with no space
    s = nltk.re.sub(r"\d", '', s)

    return s


def tokenize(x_train, y_train, x_val, y_val):
    word_list = []

    stop_words = set(stopwords.words('english'))
    for sent in x_train:
        for word in sent.lower().split():
            word = preprocess_string(word)
            if word not in stop_words and word != '':
                word_list.append(word)

    corpus = nltk.Counter(word_list)
    # sorting on the basis of most common words
    corpus_ = sorted(corpus, key=corpus.get, reverse=True)[:1000]
    # creating a dict
    onehot_dict = {w: i + 1 for i, w in enumerate(corpus_)}

    # tockenize
    final_list_train, final_list_test = [], []
    for sent in x_train:
        final_list_train.append([onehot_dict[preprocess_string(word)] for word in sent.lower().split()
                                 if preprocess_string(word) in onehot_dict.keys()])
    for sent in x_val:
        final_list_test.append([onehot_dict[preprocess_string(word)] for word in sent.lower().split()
                                if preprocess_string(word) in onehot_dict.keys()])

    encoded_train = [1 if label == 'positive' else 0 for label in y_train]
    encoded_test = [1 if label == 'positive' else 0 for label in y_val]
    return np.array(final_list_train), np.array(encoded_train), np.array(final_list_test), np.array(
        encoded_test), onehot_dict


# pad each sequence to max length
def padding(sentences, seq_len):
    features = np.zeros((len(sentences), seq_len),dtype=int)
    for ii, review in enumerate(sentences):
        if len(review) != 0:
            features[ii, -len(review):] = np.array(review)[:seq_len]
    return features


def acc(pred, label):
    pred = torch.round(pred.squeeze())
    return torch.sum(pred == label.squeeze()).item()

def predict_text(text):
    word_seq = np.array([vocab[preprocess_string(word)] for word in text.split() if preprocess_string(word) in vocab.keys()])
    word_seq = np.expand_dims(word_seq, axis=0)
    pad = torch.from_numpy(padding(word_seq, 500))
    inputs = pad.to(device)
    batch_size = 1
    h = model.init_hidden(batch_size)
    h = tuple([each.data for each in h])
    output, h = model(inputs, h)
    return (output.item())
