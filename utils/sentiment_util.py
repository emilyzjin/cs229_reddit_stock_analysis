# Utility functions for data preprocessing for deep learning sentiment analysis model.

import string
from string import punctuation
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()

def tokenize(s, stopwords=stopwords):
    """
    Given a string, remove stop words, lowercase words,
    and remove words with non-alphanumeric characters.
    """
    tokenized = s.split()
    result = []
    for word in tokenized:
        word = word.strip(punctuation)
        if word.isalpha() and word not in stopwords:
            result.append(lemmatizer.lemmatize(word).lower())
    return result