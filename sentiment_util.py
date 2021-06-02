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
    correct = 0.5 - (torch.abs(preds - label).float() / 8) + ((preds == label).float() / 2)
    # Average correct predictions
    accuracy = correct.sum() / len(correct)

    return accuracy

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

            accuracy = batch_accuracy(y, batch.label)
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
        print("calculating scores...")
        """
        precision = (tp + 1) / (tp + fp + 1)
        recall = (tp + 1) / (tp + fn + 1)
        f1 = (2 * precision * recall + 1) / (precision + recall + 1)
        mcc = (tp * tn - fp * fn + 1) / torch.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn) + 1)  
        """  
    return eval_loss / len(iterator), eval_acc / len(iterator)#, precision, recall, f1, mcc


def predict(model, text, tokenized=True):
    """
    Given a tweet, predict the sentiment.

    text - a string or a a list of tokens
    tokenized - True if text is a list of tokens, False if passing in a string
    """

    # nlp = spacy.load('en')
    nlp = spacy.blank("en")

    # Sets the model to evaluation mode
    model.eval()

    if tokenized == False:
        # Tokenizes the sentence
        tokens = [token.text for token in nlp.tokenizer(text)]
    else:
        tokens = text

    # Index the tokens by converting to the integer representation from the vocabulary
    indexed_tokens = [TEXT.vocab.stoi[t] for t in tokens]
    # Get the length of the text
    length = [len(indexed_tokens)]
    # Convert the indices to a tensor
    tensor = torch.LongTensor(indexed_tokens).to(device)
    # Add a batch dimension by unsqueezeing
    tensor = tensor.unsqueeze(1)
    # Converts the length into a tensor
    length_tensor = torch.LongTensor(length)
    # Convert prediction to be between 0 and 1 with the sigmoid function
    prediction = torch.sigmoid(model(tensor, length_tensor))

    # Return a single value from the prediction
    return prediction.item()