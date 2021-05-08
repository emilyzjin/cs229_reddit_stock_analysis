import os
import numpy as np
import random
import ujson as json
from preprocessing import get_data, write_csv
import regex as re
import util
import string


def get_tweets_change(filename):
    data = get_data(filename)
    num_data = len(data)
    tweets = [data[i][0] for i in range(num_data)]
    change = np.asarray([float(data[i][-1]) for i in range(num_data)])
    return tweets, change


def get_text_matrix(train_tweets, dev_tweets, test_tweets, train_file, dev_file, test_file, create_text_matrix, dictionary):
    if create_text_matrix:
        train_matrix = util.transform_text(train_tweets, dictionary)
        write_csv(train_matrix, train_file)
        dev_matrix = util.transform_text(dev_tweets, dictionary)
        write_csv(dev_matrix, dev_file)
        test_matrix = util.transform_text(test_tweets, dictionary)
        write_csv(test_matrix, test_file)
    else:
        train_matrix = np.asarray(get_data(train_file), dtype=float)
        dev_matrix = np.asarray(get_data(dev_file), dtype=float)
        test_matrix = np.asarray(get_data(test_file), dtype=float)

    return train_matrix, dev_matrix, test_matrix


def get_words(tweet):
    """Convert to lowercase and remove punctuation, articles and extra whitespace."""
    def remove_articles(text):
        regex = re.compile(r'\b(a|an|the|that|to|but|we|are|at|is|in|were|yourself|or|you|your|you.|me|their)\b', re.UNICODE)
        return re.sub(regex, ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    # def remove_punc(text):
    #     exclude = set(string.punctuation)
    #     return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(lower(tweet))).split()


def create_dict(tweets, num_occur):
    temp = {}
    for tweet in tweets:
        words = set(get_words(tweet))
        for word in words:
            count = temp.get(word, 0) + 1
            temp[word] = count
    temp = {key: value for (key, value) in temp.items() if value >= num_occur}
    dict = {index: word for index, word in enumerate(temp)}
    return dict


def data_iter(batch_size, x, y):
    num_examples = len(x)
    indices = list(range(num_examples))
    random.shuffle(indices)
    for i in range(0, num_examples, batch_size):
        batch_indices = np.array(indices[i:min(i + batch_size, num_examples)])
        yield x[batch_indices], y[batch_indices]


def transform_text(tweets, dictionary):
        """Transform a list of text messages into a numpy array that contains the number of
        times each word of the vocabulary appears in each message.
        """
        n, d = len(tweets), len(dictionary)
        occur = np.zeros((n, d))

        for i in range(n):
            if i % 1000 == 0:
                print('Sample', i, 'out of', n)
            words = get_words(tweets[i])
            for index, word in dictionary.items():
                occur[i, index] = words.count(word)
        return occur


def write_json(filename, value):
    """Write the provided value as JSON to the given filename"""
    with open(filename, 'w') as f:
        json.dump(value, f)

def get_top_words(num_words, tweet_matrix, change, dict):
    """Returns list of words with highest expected percent change in price (in increasing order), given the word appears in a message
       num_words = number of top words to get
       tweet_matrix = np array where (i,j) entry corresponds to number
       of times the j-th word in the dictionary appears in the i-th tweet
       change = percent change in price
       dict = dictionary where key = column in tweet_matrix, value = word
    """
    expected = np.sum(tweet_matrix * change[:, np.newaxis], axis=0)
    top_words = np.argsort(expected)[-num_words:]
    top_words = [dict.get(i) for i in reversed(top_words)]
    return top_words

