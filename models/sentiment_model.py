# Sentiment Analysis Model from https://towardsdatascience.com/sentiment-analysis-using-lstm-step-by-step-50d074f09948

from utils.sentiment_util import tokenize
import torch.nn as nn


class SentimentAnalysis:
    def __init__(self, vocab_size, output_size, embedding_dim, hidden_dim, num_layers, drop_prob=0.5):
        """
        Initialize the model by setting up the layers.
        """
        self.output_size = output_size
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim

        # embedding and LSTM layers
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers,
                            dropout=drop_prob, batch_first=True)

        # dropout layer
        self.dropout = nn.Dropout(0.3)

        # linear and sigmoid layers
        self.fc = nn.Linear(hidden_dim, output_size)
        self.sig = nn.Sigmoid()

    def forward(self, input, hidden, stopwords=None):
        """
        Perform a forward pass given input and hidden state.

        STEPS
        0. Tokenize
        1. Embedding Layer: that converts our word tokens (integers) into embedding of specific size
        2. LSTM Layer: defined by hidden state dims and number of layers
        3. Dropout
        4. Fully Connected Layer: that maps output of LSTM layer to a desired output size
        5. Sigmoid Activation Layer: that turns all output values in a value between 0 and 1
        6. Output: Sigmoid output from the last timestep is considered as the final output of this network
        """
        batch_size = input.size(0)

        # tokenize
        if stopwords is None:
            input = tokenize(input)
        else:
            input = tokenize(input, stopwords)

        # embedding and lstm
        embeddings = self.embedding(input)
        lstm_output, hidden = self.lstm(embeddings, hidden)
        lstm_output = lstm_output.contiguous().view(-1, self.hidden_dim)

        # dropout, fully-connected, sigmoid
        output = self.dropout(lstm_output)
        output = self.fc(output)
        output = self.sig(output)

        # reshape to be batch_size first
        output = output.view(batch_size, -1)
        output = output[:, -1]  # get last batch of labels

        # return last sigmoid output and hidden state
        return output, hidden

    def init_hidden(self, batch_size):
        """
        Initializes hidden state
        """
        # Create two new tensors with sizes n_layers x batch_size x hidden_dim,
        # initialized to zero, for hidden state and cell state of LSTM
        weight = next(self.parameters()).data

        if (train_on_gpu):
            hidden = (weight.new(self.num_layers, batch_size, self.hidden_dim).zero_().cuda(),
                      weight.new(self.num_layers, batch_size, self.hidden_dim).zero_().cuda())
        else:
            hidden = (weight.new(self.num_layers, batch_size, self.hidden_dim).zero_(),
                      weight.new(self.num_layers, batch_size, self.hidden_dim).zero_())

        return hidden
