import torch
import torch.nn as nn
import torchtext
import csv
from models.sentiment_model import SentimentLSTM
from utils.sentiment_util import evaluate
from torchtext.legacy import data
import spacy
from torchtext.vocab import GloVe


# def data_preprocess(csv_file):
#     data = torch.tensor(())
#     glove = GloVe(cache='.', name='6B')
#     with open(csv_file) as f:
#         reader = csv.reader(f, delimiter=',')
#         for row in reader:
#             # Convert data into word embeddings (PyTorch tensor).
#             text = row[:-4][0]
#             text = text.split(', ')
#             embed = []
#             for word in text:
#                 word = glove[word]
#                 embed.append(word)
#
#             # max_vocab_size = 287799
#
#             # TEXT -> WORD EMBEDDING
#             word_embedding = None
#             row_data = torch.cat((word_embedding, torch.tensor(row[-3:])), dim=0)
#             data = torch.cat((data, row_data), dim=0)
#     f.close()
#     pass
#
#     # data.shape = (N, word_embedding_size + 3 (1 for upvotes - downvotes, 1 for change in the last 7 days, label (next 7 days)))
#     idxs = torch.randperm(data.shape[0])
#     data = data[idxs, :]
#     train_size = data.shape[0] * 0.8
#     val_size = data.shape[0] * 0.1
#     test_size = data.shape[0] - train_size - val_size
#     train_split, val_split, test_split = torch.split(data, [train_size, val_size, test_size], dim=0)
#
#     return train_split, val_split, test_split


def create_csv():
    with open('removed_characters.csv') as in_file:
        with open('data_text.csv', 'w') as text_file:
            with open('data_other.csv', 'w') as other_file:
                reader = csv.reader(in_file, delimiter=',')
                writer_1 = csv.writer(text_file)
                writer_2 = csv.writer(other_file)
                for row in reader:
                    # text = row[:-4]
                    text = row[0].split(', ')
                    text = ' '.join(text)
                    text = [text]
                    other = row[-3:]
                    writer_1.writerow(text)
                    writer_2.writerow(other)
    in_file.close()


def data_preprocess(max_vocab_size, device, batch_size):
    spacy.load("en_core_web_sm")

    TEXT = data.Field(tokenize='spacy', lower=True, include_lengths=True)
    LABEL = data.LabelField(dtype=torch.float)

    # Map data to fields
    fields = [('text', TEXT)]

    # Apply field definition to create torch dataset
    dataset = data.TabularDataset(
        path="data_text.csv",
        format="CSV",
        fields=fields,
        skip_header=False)

    # Split data into train, test, validation sets
    (train_data, test_data, valid_data) = dataset.split(split_ratio=[0.8, 0.1, 0.1])

    print("Number of train data: {}".format(len(train_data)))
    print("Number of test data: {}".format(len(test_data)))
    print("Number of validation data: {}".format(len(valid_data)))

    # unk_init initializes words in the vocab using the Gaussian distribution
    TEXT.build_vocab(train_data,
                     max_size=max_vocab_size,
                     vectors="glove.6B.100d")

    # build vocab - convert words into integers
    LABEL.build_vocab(train_data)

    train_iterator, valid_iterator, test_iterator = data.BucketIterator.splits(
        (train_data, valid_data, test_data),
        device=device,
        batch_size=batch_size,
        sort_key=lambda x: len(x.text),
        sort_within_batch=True)

    return train_iterator, valid_iterator, test_iterator


def train():
    # run train_iterator into the
    pass

def test():
    pass


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    batch_size = 128
    create_csv()
    train_iterator, valid_iterator, test_iterator = data_preprocess(25000, device, batch_size)

    print(1)


if __name__=="__main__":
    main()