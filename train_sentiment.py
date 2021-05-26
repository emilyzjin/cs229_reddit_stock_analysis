# Train Loop from https://towardsdatascience.com/sentiment-analysis-using-lstm-step-by-step-50d074f09948
# TODO: define vocab_to_int, batch_size, valid_loader, train_on_gpu

import torch
import torch.nn as nn
from models.sentiment_model import SentimentAnalysis
import numpy as np


def main():
    # Instantiate the model w/ hyperparams
    vocab_size = len(vocab_to_int) + 1  # +1 for the 0 padding
    output_size = 1
    embedding_dim = 400
    hidden_dim = 256
    num_layers = 2
    drop_prob = 0.5
    model = SentimentAnalysis(vocab_size, output_size, embedding_dim, hidden_dim, num_layers, drop_prob)

    lr = 0.001

    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # training params
    epochs = 4  # 3-4 is approx where I noticed the validation loss stop decreasing

    counter = 0
    print_every = 100
    clip = 5  # gradient clipping

    # move model to GPU, if available
    if (train_on_gpu):
        model.cuda()

    model.train()
    # train for some number of epochs
    for e in range(epochs):
        # initialize hidden state
        h = model.init_hidden(batch_size)

        # batch loop
        for inputs, labels in train_loader:
            counter += 1

            if (train_on_gpu):
                inputs, labels = inputs.cuda(), labels.cuda()

            # Creating new variables for the hidden state, otherwise
            # we'd backprop through the entire training history
            h = tuple([each.data for each in h])

            model.zero_grad()

            # get the output from the model
            inputs = inputs.type(torch.LongTensor)
            output, h = model(inputs, h)

            # calculate the loss and perform backprop
            loss = criterion(output.squeeze(), labels.float())
            loss.backward()
            # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
            nn.utils.clip_grad_norm_(model.parameters(), clip)
            optimizer.step()

            # loss stats
            if counter % print_every == 0:
                # Get validation loss
                val_h = model.init_hidden(batch_size)
                val_losses = []
                model.eval()
                for inputs, labels in valid_loader:

                    # Creating new variables for the hidden state, otherwise
                    # we'd backprop through the entire training history
                    val_h = tuple([each.data for each in val_h])

                    if (train_on_gpu):
                        inputs, labels = inputs.cuda(), labels.cuda()

                    inputs = inputs.type(torch.LongTensor)
                    output, val_h = model(inputs, val_h)
                    val_loss = criterion(output.squeeze(), labels.float())

                    val_losses.append(val_loss.item())

                model.train()
                print("Epoch: {}/{}...".format(e + 1, epochs),
                      "Step: {}...".format(counter),
                      "Loss: {:.6f}...".format(loss.item()),
                      "Val Loss: {:.6f}".format(np.mean(val_losses)))

if __name__ == "__main__":
    main()