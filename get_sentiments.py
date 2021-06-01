import csv
from csv import reader
from csv import writer
import string
from string import punctuation
from sentiment_util import tokenize
import nltk
nltk.download('wordnet')
nltk.download('stopwords')
nltk.download('vader_lexicon')
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import torch
import torch.nn as nn

with open('removed_characters_buckets.csv', 'r', encoding='utf-8') as f:
    reader = reader(f)
    rows = []
    for row in reader:
        rows.append(row)
f.close()

with open('removed_characters_buckets_sentiments.csv', 'w', encoding='utf-8') as f:
    writer = writer(f)
    for row in rows:
        analyzer = SentimentIntensityAnalyzer()
        row[0] = tokenize(row[0])
        scores = analyzer.polarity_scores(row[0])
        row.insert(-1, scores['neg'])
        row.insert(-1, scores['neu'])
        row.insert(-1, scores['pos'])
        row.insert(-1, scores['compound'])
        writer.writerow(row)
f.close()
print("done")

