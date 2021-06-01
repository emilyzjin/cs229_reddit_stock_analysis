from matplotlib.pyplot import uninstall_repl_displayhook
import torch
import torch.nn as nn
import torchtext
import csv 
from util import *
from models.sentiment_model import MovementPredictor
from sentiment_util import evaluate
from torchtext.legacy import data
import spacy
import torch.optim as optim
import torch.optim.lr_scheduler as sched
from torchtext.vocab import GloVe
import torch.nn.functional as F
import pdb


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
 

def test():
    pass


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train = True
    batch_size = 128
    hidden_size = 200
    drop_prob = 0.2
    learning_rate = 1e-3
    num_epochs = 100
    save_dir = 'models/model.path.tar' # TODO: SET PATH.
    eval_interval = 3
    beta1, beta2 = 0.9, 0.999 # for Adam
    alpha = 0.2 # for ELU
    max_grad_norm = 1.0 # TODO: ?? WHAT IS THIS
    # create_csv()

    # Initialize model.
    model = MovementPredictor(
        vocab_size=287799, # TODO
        embedding_dim=100, # TODO
        hidden_dim=hidden_size,
        output_dim=5, # TODO
        n_layers=1, # TODO
        bidirectional=True,
        dropout=drop_prob,
        pad_idx=None, # TODO
        alpha=0.2
    )
    device, gpu_ids = util.get_available_devices()
    model = nn.DataParallel(model, gpu_ids)

    # Initialize optimizer and scheduler.
    optimizer = optim.Adam(model.parameters, lr=learning_rate, betas=(beta1, beta2))
    #scheduler = sched.LambdaLR(optimizer, lambda s: 1.)

    train_iterator, valid_iterator, test_iterator = data_preprocess(25000, device, batch_size)
    # Training Loop
    if train:
        steps_till_eval = eval_interval
        for epoch in range(num_epochs):
            steps_till_eval -= 1
            with torch.enable_grad():
                # TODO: maybe we should split data and then use dataloader here?
                for vector in train_iterator:
                    # Grab labels.
                    target = torch.zeros((5,))
                    target[train_iterator[:, -1]] = 1
                    # Grab other data for multimodal sentiment analysis.
                    multimodal_data = train_iterator[:, -3:-2] # Upvotes + past week change
                    # Apply model
                    y = model(vector[:, :-4], multimodal_data)
                    target = target.to(device) # TODO: Unsure if this line is needed?
                    loss = F.BCELoss(y, target)
                    loss_val = loss.item()

                    # Backward
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                    optimizer.step()
                    #scheduler.step(step // batch_size)
                    # TODO: Print + Log (not sure if needed rn)
                torch.save(model, save_dir)    
                if steps_till_eval == 0:
                    print("evaluating on dev split...")
                    loss, accuracy = evaluate(model, data=valid_iterator, criterion=nn.BCEWithLogitsLoss())
                    print("dev loss: ", loss, "dev accuracy: ", accuracy)
                    steps_till_eval = 3
                
    else: 
        # testing case
        print("testing data, loading from path" + save_dir + " ...")
        model = torch.load(save_dir)
        loss, accuracy = evaluate(model, test_iterator, criterion=nn.BCEWithLogitsLoss())
        print("test loss: ", loss, "test accuracy: ", accuracy)
    pdb.set_trace()
    print(1)


if __name__=="__main__":
    main()