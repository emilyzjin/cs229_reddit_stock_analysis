# Utility functions for data preprocessing for deep learning sentiment analysis model.

import csv
import string
from string import punctuation
import nltk
nltk.download('wordnet')
nltk.download('stopwords')
nltk.download('vader_lexicon')
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import torch
import torch.nn as nn
from evaluate import *

lemmatizer = WordNetLemmatizer()


def tokenize(s, stopwords=stopwords):
    """
    Given a string, remove stop words, lowercase words,
    and remove words with non-alphanumeric characters.
    """
    tokenized = s.split()
    result = ""
    for word in tokenized:
        word = word.strip(punctuation)
        if word.isalpha() and word not in stopwords.words('english'):
            result += lemmatizer.lemmatize(word).lower()
            result += " "
    return result


def tokenize_csv(input_file, output_file):
    with open(input_file, 'r') as f:
        reader = csv.reader(f)
        dataset_list = list(reader)
        tokenized_data = []
        i=0
        for entry in dataset_list:
            tokenized_entry = tokenize(entry[-1])
            if len(tokenized_entry) > 0:
                tokenized_data.append([entry[0], tokenized_entry])
                i += 1 
                if i % 20000 == 0:
                    print(i / len(dataset_list))
            
    with open(output_file, 'w', newline='') as f:
        write = csv.writer(f)
        for entry in tokenized_data:
            write.writerow(entry)

    print('done')


def batch_accuracy(predictions, label):
    """
    Returns accuracy per batch.

    predictions - float
    """

    # Round predictions to the closest integer using the sigmoid function
    preds = torch.argmax(predictions, dim=1)
    # If prediction is equal to label
    close = 0.5 - (torch.abs(preds - label).float() / 8) + ((preds == label).float() / 2)
    correct = (preds == label).float()
    # Average correct predictions
    accuracy = correct.sum() / len(correct)
    closeness = close.sum() / len(correct)
    return accuracy, closeness

def timer(start_time, end_time):
    """
    Returns the minutes and seconds.
    """

    time = end_time - start_time
    mins = int(time / 60)
    secs = int(time - (mins * 60))

    return mins, secs

def evaluate(model, iterator, device):
    """
    Function to evaluate the loss and accuracy of validation and test sets.
    iterator - validation or test iterator
    """
    
    # Cumulated Training loss
    eval_loss = 0.0
    # Cumulated Training accuracy
    eval_acc = 0
    eval_closeness = 0
    tp, fp, tn, fn = 0, 0, 0, 0
            
    # Don't calculate the gradients
    with torch.no_grad():
        for batch in iterator:
            # Grab labels.
            #target = torch.zeros((batch.batch_size, 5))
            target = batch.label.type(dtype=torch.int64)
            #target[torch.arange(batch.batch_size), batch.label.type(dtype=torch.int64)] = 1
            # Grab other data for multimodal sentiment analysis.
            multimodal_data = torch.cat((batch.upvote.unsqueeze(dim=1),
                                         batch.change.unsqueeze(dim=1), 
                                         batch.sent.unsqueeze(dim=1)), dim=1)  # Upvotes + past week change
            # Apply model
            y = model(batch, multimodal_data)
            target = target.to(device)
            loss_function = nn.CrossEntropyLoss()
            loss = loss_function(y, target)

            accuracy, closeness = batch_accuracy(y, batch.label)
            """
            for i in range(5):
                preds_binary = torch.where(y == i, 1, 0)
                labels_binary = torch.where(batch.label == i, 1, 0)
                tpi, fpi, tni, fni = calc_numbers(preds_binary, labels_binary)
                tp += tpi
                fp += fpi 
                tn += tni 
                fn += fni
            """
            eval_loss += loss.item()
            eval_acc += accuracy.item()
            eval_closeness += closeness.item()
        print("calculating scores...")
        """
        precision = (tp + 1) / (tp + fp + 1)
        recall = (tp + 1) / (tp + fn + 1)
        f1 = (2 * precision * recall + 1) / (precision + recall + 1)
        mcc = (tp * tn - fp * fn + 1) / torch.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn) + 1)  
        """  
    return eval_loss / len(iterator), eval_acc / len(iterator), eval_closeness / len(iterator)#, precision, recall, f1, mcc


def create_csv():
    with open('removed_characters.csv') as in_file:
        with open('removed_characters_buckets.csv', 'w') as out_file:
            reader = csv.reader(in_file, delimiter=',')
            writer = csv.writer(out_file)
            for row in reader:
                text = row[0].split(', ')
                text = ' '.join(text)
                row_data = [text]
                row_data.extend(row[-3:-1])
                label = 1 - float(row[-1])
                # Strong buy
                if label >= .03:
                    label = 0
                # Buy
                elif .01 < label < .03:
                    label = 1
                # Hold
                elif -.01 <= label <= .01:
                    label = 2
                # Sell
                elif -.01 > label > -.03:
                    label = 3
                else:
                    label = 4
                row_data.append(label)
                writer.writerow(row_data)
    in_file.close()


def data_preprocess(TEXT, UPVOTE, CHANGE, SENT, LABEL, data, max_vocab_size, device, batch_size):

    # Map data to fields
    fields_text = [('text', TEXT), ('upvote', UPVOTE), ('change', CHANGE), ('sent', SENT), ('label', LABEL)]

    # Apply field definition to create torch dataset
    train_data = data.TabularDataset(
        path="train_data.csv",
        format="CSV",
        fields=fields_text,
        skip_header=False)
    valid_data = data.TabularDataset(
        path="valid_data.csv",
        format="CSV",
        fields=fields_text,
        skip_header=False)

    test_data = data.TabularDataset(
        path="valid_data.csv",
        format="CSV",
        fields=fields_text,
        skip_header=False)

    print("Number of train data: {}".format(len(train_data)))
    print("Number of test data: {}".format(len(test_data)))
    print("Number of validation data: {}".format(len(valid_data)))

    # unk_init initializes words in the vocab using the Gaussian distribution
    TEXT.build_vocab(train_data,
                     max_size=max_vocab_size,
                     vectors="glove.6B.100d",
                     unk_init=torch.Tensor.normal_)

    train_iterator, valid_iterator, test_iterator = data.BucketIterator.splits(
        (train_data, valid_data, test_data),
        device=device,
        batch_sizes=(batch_size, batch_size, batch_size),
        sort_key=lambda x: len(x.text),
        sort_within_batch=False)

    return train_iterator, valid_iterator, test_iterator
