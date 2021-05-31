# Utility functions for data preprocessing for deep learning sentiment analysis model.

import csv
import string
from string import punctuation
import nltk
nltk.download('wordnet')
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import torch
import spacy
spacy.load("en_core_web_sm")

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
    label - 0 or 1
    """

    # Round predictions to the closest integer using the sigmoid function
    preds = torch.round(torch.sigmoid(predictions))
    # If prediction is equal to label
    correct = 1 - (torch.abs(preds - label).float() / 4)
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


def train(model, iterator, optimizer, criterion):
    """
    Function to evaluate training loss and accuracy.

    iterator - train iterator
    """

    # Cumulated Training loss
    training_loss = 0.0
    # Cumulated Training accuracy
    training_acc = 0.0

    # Set model to training mode
    model.train()

    # For each batch in the training iterator
    for batch in iterator:

        # 1. Zero the gradients
        optimizer.zero_grad()

        # batch.text is a tuple (tensor, len of seq)
        text, text_lengths = batch.text

        # 2. Compute the predictions
        predictions = model(text, text_lengths).squeeze(1)

        # 3. Compute loss
        loss = criterion(predictions, batch.label)

        # Compute accuracy
        accuracy = batch_accuracy(predictions, batch.label)

        # 4. Use loss to compute gradients
        loss.backward()

        # 5. Use optimizer to take gradient step
        optimizer.step()

        training_loss += loss.item()
        training_acc += accuracy.item()

    # Return the loss and accuracy, averaged across each epoch
    # len of iterator = num of batches in the iterator
    return training_loss / len(iterator), training_acc / len(iterator)

def evaluate(model, iterator, criterion):
    """
    Function to evaluate the loss and accuracy of validation and test sets.

    iterator - validation or test iterator
    """

    # Cumulated Training loss
    eval_loss = 0.0
    # Cumulated Training accuracy
    eval_acc = 0

    # Set model to evaluation mode
    model.eval()

    # Don't calculate the gradients
    with torch.no_grad():

        for batch in iterator:

            text, text_lengths = batch.text

            predictions = model(text, text_lengths).squeeze(1)

            loss = criterion(predictions, batch.label)

            accuracy = batch_accuracy(predictions, batch.label)

            eval_loss += loss.item()
            eval_acc += accuracy.item()

    return eval_loss / len(iterator), eval_acc / len(iterator)


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