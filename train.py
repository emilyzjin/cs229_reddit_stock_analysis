import torch
import csv

def data_preprocess(csv_file):
    data = torch.tensor(())
    with open(csv_file) as f:
        reader = csv.reader(f, delimiter=',')
        for row in reader:
            # Convert data into word embeddings (PyTorch tensor).
            text = row[:-4]
            # TEXT -> WORD EMBEDDING
            word_embedding = None
            row_data = torch.cat((word_embedding, torch.tensor(row[-3:])), dim=0)
            data = torch.cat((data, row_data), dim=0)
    f.close()
    pass

    # data.shape = (N, word_embedding_size + 3 (1 for upvotes - downvotes, 1 for change in the last 7 days, label (next 7 days)))
    idxs = torch.randperm(data.shape[0])
    data = data[idxs, :]
    train_size = data.shape[0] * 0.8
    val_size = data.shape[0] * 0.1
    test_size = data.shape[0] - train_size - val_size
    train_split, val_split, test_split = torch.split(data, [train_size, val_size, test_size], dim=0)

    return train_split, val_split, test_split

def train():
    pass

def test():
    pass