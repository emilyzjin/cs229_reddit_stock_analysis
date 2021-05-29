# Train Loop from https://towardsdatascience.com/sentiment-analysis-using-lstm-step-by-step-50d074f09948
# TODO: define vocab_to_int, batch_size, valid_loader, train_on_gpu

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import torch
import torch.nn as nn
import torch.nn.functional as F
from nltk.corpus import stopwords
from collections import Counter
import string
import re
import seaborn as sns
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from utils.sentiment_util import preprocess_string, tokenize, padding, acc
from models.sentiment_model import SentimentRNN

# def main():
#     # read data from text files
#     with open(‘data / reviews.txt’, ‘r’) as f:
#         reviews = f.read()
#     with open(‘data / labels.txt’, ‘r’) as f:
#         labels = f.read()
#
#     # data processing - convert to lower
#     reviews = reviews.lower()
#     all_text = ''.join([c for c in reviews if c not in punctuation])
#
#     # Instantiate the model w/ hyperparams
#     vocab_size = len(vocab_to_int) + 1  # +1 for the 0 padding
#     output_size = 1
#     embedding_dim = 400
#     hidden_dim = 256
#     num_layers = 2
#     drop_prob = 0.5
#     model = SentimentAnalysis(vocab_size, output_size, embedding_dim, hidden_dim, num_layers, drop_prob)
#
#     lr = 0.001
#
#     criterion = nn.BCELoss()
#     optimizer = torch.optim.Adam(model.parameters(), lr=lr)
#
#     # training params
#     epochs = 4  # 3-4 is approx where I noticed the validation loss stop decreasing
#
#     counter = 0
#     print_every = 100
#     clip = 5  # gradient clipping
#
#     # move model to GPU, if available
#     if (train_on_gpu):
#         model.cuda()
#
#     model.train()
#     # train for some number of epochs
#     for e in range(epochs):
#         # initialize hidden state
#         h = model.init_hidden(batch_size)
#
#         # batch loop
#         for inputs, labels in train_loader:
#             counter += 1
#
#             if (train_on_gpu):
#                 inputs, labels = inputs.cuda(), labels.cuda()
#
#             # Creating new variables for the hidden state, otherwise
#             # we'd backprop through the entire training history
#             h = tuple([each.data for each in h])
#
#             model.zero_grad()
#
#             # get the output from the model
#             inputs = inputs.type(torch.LongTensor)
#             output, h = model(inputs, h)
#
#             # calculate the loss and perform backprop
#             loss = criterion(output.squeeze(), labels.float())
#             loss.backward()
#             # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
#             nn.utils.clip_grad_norm_(model.parameters(), clip)
#             optimizer.step()
#
#             # loss stats
#             if counter % print_every == 0:
#                 # Get validation loss
#                 val_h = model.init_hidden(batch_size)
#                 val_losses = []
#                 model.eval()
#                 for inputs, labels in valid_loader:
#
#                     # Creating new variables for the hidden state, otherwise
#                     # we'd backprop through the entire training history
#                     val_h = tuple([each.data for each in val_h])
#
#                     if (train_on_gpu):
#                         inputs, labels = inputs.cuda(), labels.cuda()
#
#                     inputs = inputs.type(torch.LongTensor)
#                     output, val_h = model(inputs, val_h)
#                     val_loss = criterion(output.squeeze(), labels.float())
#
#                     val_losses.append(val_loss.item())
#
#                 model.train()
#                 print("Epoch: {}/{}...".format(e + 1, epochs),
#                       "Step: {}...".format(counter),
#                       "Loss: {:.6f}...".format(loss.item()),
#                       "Val Loss: {:.6f}".format(np.mean(val_losses)))


def main():
    is_cuda = torch.cuda.is_available()

    # If we have a GPU available, we'll set our device to GPU. We'll use this device variable later in our code.
    if is_cuda:
        device = torch.device("cuda")
        print("GPU is available")
    else:
        device = torch.device("cpu")
        print("GPU not available, CPU used")

    base_csv = '/kaggle/input/imdb-dataset-of-50k-movie-reviews/IMDB Dataset.csv'
    df = pd.read_csv(base_csv)
    df.head()

    X, y = df['review'].values, df['sentiment'].values
    x_train, x_test, y_train, y_test = train_test_split(X, y, stratify=y)
    print(f'shape of train data is {x_train.shape}')
    print(f'shape of test data is {x_test.shape}')

    dd = pd.Series(y_train).value_counts()
    sns.barplot(x=np.array(['negative', 'positive']), y=dd.values)
    plt.show()

    # tokenize
    x_train, y_train, x_test, y_test, vocab = tokenize(x_train, y_train, x_test, y_test)

    # padding
    x_train_pad = padding(x_train, 500)
    x_test_pad = padding(x_test, 500)

    # create Tensor datasets
    train_data = TensorDataset(torch.from_numpy(x_train_pad), torch.from_numpy(y_train))
    valid_data = TensorDataset(torch.from_numpy(x_test_pad), torch.from_numpy(y_test))

    # dataloaders
    batch_size = 50

    # make sure to SHUFFLE your data
    train_loader = DataLoader(train_data, shuffle=True, batch_size=batch_size)
    valid_loader = DataLoader(valid_data, shuffle=True, batch_size=batch_size)

    # obtain one batch of training data
    dataiter = iter(train_loader)
    sample_x, sample_y = dataiter.next()

    print('Sample input size: ', sample_x.size())  # batch_size, seq_length
    print('Sample input: \n', sample_x)
    print('Sample input: \n', sample_y)

    # initialize hyperparameters
    no_layers = 2
    vocab_size = len(vocab) + 1  # extra 1 for padding
    embedding_dim = 64
    output_dim = 1
    hidden_dim = 256

    model = SentimentRNN(no_layers, vocab_size, hidden_dim, embedding_dim, output_dim, drop_prob=0.5)

    # moving to gpu
    model.to(device)

    print(model)

    # Training
    # loss and optimization functions
    lr = 0.001
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    clip = 5
    epochs = 5
    valid_loss_min = np.Inf
    # train for some number of epochs
    epoch_tr_loss, epoch_vl_loss = [], []
    epoch_tr_acc, epoch_vl_acc = [], []

    for epoch in range(epochs):
        train_losses = []
        train_acc = 0.0
        model.train()
        # initialize hidden state
        h = model.init_hidden(batch_size)
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            # Creating new variables for the hidden state, otherwise
            # we'd backprop through the entire training history
            h = tuple([each.data for each in h])

            model.zero_grad()
            output, h = model(inputs, h)

            # calculate the loss and perform backprop
            loss = criterion(output.squeeze(), labels.float())
            loss.backward()
            train_losses.append(loss.item())
            # calculating accuracy
            accuracy = acc(output, labels)
            train_acc += accuracy
            # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
            nn.utils.clip_grad_norm_(model.parameters(), clip)
            optimizer.step()

        val_h = model.init_hidden(batch_size)
        val_losses = []
        val_acc = 0.0
        model.eval()
        for inputs, labels in valid_loader:
            val_h = tuple([each.data for each in val_h])

            inputs, labels = inputs.to(device), labels.to(device)

            output, val_h = model(inputs, val_h)
            val_loss = criterion(output.squeeze(), labels.float())

            val_losses.append(val_loss.item())

            accuracy = acc(output, labels)
            val_acc += accuracy

        epoch_train_loss = np.mean(train_losses)
        epoch_val_loss = np.mean(val_losses)
        epoch_train_acc = train_acc / len(train_loader.dataset)
        epoch_val_acc = val_acc / len(valid_loader.dataset)
        epoch_tr_loss.append(epoch_train_loss)
        epoch_vl_loss.append(epoch_val_loss)
        epoch_tr_acc.append(epoch_train_acc)
        epoch_vl_acc.append(epoch_val_acc)
        print(f'Epoch {epoch + 1}')
        print(f'train_loss : {epoch_train_loss} val_loss : {epoch_val_loss}')
        print(f'train_accuracy : {epoch_train_acc * 100} val_accuracy : {epoch_val_acc * 100}')
        if epoch_val_loss <= valid_loss_min:
            torch.save(model.state_dict(), '../working/state_dict.pt')
            print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(valid_loss_min,
                                                                                            epoch_val_loss))
            valid_loss_min = epoch_val_loss
        print(25 * '==')

        fig = plt.figure(figsize=(20, 6))
        plt.subplot(1, 2, 1)
        plt.plot(epoch_tr_acc, label='Train Acc')
        plt.plot(epoch_vl_acc, label='Validation Acc')
        plt.title("Accuracy")
        plt.legend()
        plt.grid()

        plt.subplot(1, 2, 2)
        plt.plot(epoch_tr_loss, label='Train loss')
        plt.plot(epoch_vl_loss, label='Validation loss')
        plt.title("Loss")
        plt.legend()
        plt.grid()

        plt.show()

        def predict_text(text):
            word_seq = np.array(
                [vocab[preprocess_string(word)] for word in text.split() if preprocess_string(word) in vocab.keys()])
            word_seq = np.expand_dims(word_seq, axis=0)
            pad = torch.from_numpy(padding(word_seq, 500))
            inputs = pad.to(device)
            batch_size = 1
            h = model.init_hidden(batch_size)
            h = tuple([each.data for each in h])
            output, h = model(inputs, h)
            return (output.item())

        index = 30
        print(df['review'][index])
        print('=' * 70)
        print(f'Actual sentiment is  : {df["sentiment"][index]}')
        print('=' * 70)
        pro = predict_text(df['review'][index])
        status = "positive" if pro > 0.5 else "negative"
        pro = (1 - pro) if status == "negative" else pro
        print(f'Predicted sentiment is {status} with a probability of {pro}')
if __name__ == "__main__":
    main()