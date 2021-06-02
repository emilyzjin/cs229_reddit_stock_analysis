import torch
import torch.nn as nn
import torchtext
import csv
from util import get_available_devices
from sentiment_util import evaluate
from models.sentiment_model import MovementPredictor, WithoutSentiment, WithSentiment
from torchtext.legacy import data
import spacy
import torch.optim as optim
import torch.optim.lr_scheduler as sched
from torchtext.vocab import GloVe
import torch.nn.functional as F
import pdb

# spacy.load("en_core_web_sm")
# spacy_en = spacy.load('en_core_web_sm')
spacy.load('en', disable=['ner', 'parser', 'tagger'])

def tokenize(s):
    return s.split(' ')

TEXT = data.Field(tokenize=tokenize, lower=True, include_lengths=True)
UPVOTE = data.LabelField(sequential=False, use_vocab=False, dtype=torch.int64)
CHANGE = data.LabelField(sequential=False, use_vocab=False, dtype=torch.float)
SENT = data.LabelField(sequential=False, use_vocab=False, dtype=torch.int64)
LABEL = data.LabelField(sequential=False, use_vocab=False, dtype=torch.int64)


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


def data_preprocess(max_vocab_size, device, batch_size):

    # Map data to fields
    fields_text = [('text', TEXT), ('upvote', UPVOTE), ('change', CHANGE), ('sent', SENT), ('label', LABEL)]

    # Apply field definition to create torch dataset
    dataset = data.TabularDataset(
        path="reddit_sentiments.csv",
        format="CSV",
        fields=fields_text,
        skip_header=False)

    # Split data into train, test, validation sets
    (train_data, test_data, valid_data) = dataset.split(split_ratio=[0.8, 0.1, 0.1])

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


def main():
    create_csv()
    train = True
    batch_size = 2048
    hidden_size = 256
    output_dim = 1
    drop_prob = 0.3
    learning_rate = 1e-3 # TODO: hyper
    num_epochs = 100
    beta1, beta2 = 0.9, 0.999 # for Adam
    alpha = 0.2 # for ELU # TODO: hyper
    max_grad_norm = 2.0
    print_every = 100
    train_sentiment = False
    use_sentiment = True
    save_dir = 'results/model.path_lr_{:.4}_drop_prob_{:.4}_alpha_{:.4}.tar'.format(learning_rate, drop_prob, alpha)

    device, gpu_ids = get_available_devices()

    train_iterator, valid_iterator, test_iterator = data_preprocess(25000, device, batch_size)

    # Initialize model.
    if train_sentiment:
        model = MovementPredictor(
            vocab_size=len(TEXT.vocab),
            embedding_dim=100,
            hidden_dim=hidden_size,
            output_dim=output_dim,
            n_layers=2,
            bidirectional=True,
            dropout=drop_prob,
            pad_idx=TEXT.vocab.stoi[TEXT.pad_token],
            alpha=alpha
        )
    elif use_sentiment:
        model = WithSentiment(
            hidden_dim=hidden_size,
            alpha=alpha
        )
    else:
        model = WithoutSentiment(
            hidden_dim=hidden_size,
            alpha=alpha
        )

    # pretrained_embeddings = TEXT.vocab.vectors
    # model.embedding.weight.data.copy_(pretrained_embeddings)

    model = nn.DataParallel(model, gpu_ids)

    # Initialize optimizer and scheduler.
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, betas=(beta1, beta2))
    #scheduler = sched.LambdaLR(optimizer, lambda s: 1.)

    iter = 0
    checkpoint = 1

    # Training Loop
    if train:
        for epoch in range(num_epochs):
            iter = 0
            with torch.enable_grad():
                for vector in train_iterator:
                    optimizer.zero_grad()
                    # Grab labels.
                    #target = torch.zeros((batch_size, 5))
                    #target[torch.arange(batch_size), vector.label] = 1
                    target = vector.label
                    # Grab other data for multimodal sentiment analysis.
                    multimodal_data = torch.cat((vector.upvote.unsqueeze(dim=1), # upvotes
                                                 vector.change.unsqueeze(dim=1), # past week change
                                                 vector.sent.unsqueeze(dim=1)), # sentiment
                                                 dim=1)
                    # Apply model
                    y = model(vector, multimodal_data)
                    target = target.to(device)
                    loss_function = nn.CrossEntropyLoss()
                    loss = loss_function(y, target)
                    loss_val = loss.item()

                    # Backward
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                    optimizer.step()
                    #scheduler.step(step // batch_size)
                    if iter % print_every == 0:
                        print('Epoch:{}, Iter: {}, Loss:{:.4}'.format(epoch, iter, loss.item()))
                    iter += 1

                torch.save(model, save_dir)
                if checkpoint % 3 == 0:
                    print("evaluating on dev split...")
                    # loss_val, accuracy, precision, recall, f1, mcc = evaluate(model, valid_iterator, device)
                    loss_val, accuracy, closeness = evaluate(model, valid_iterator, device)
                    with open('results/model.path_lr_{:.4}_drop_prob_{:.4}_alpha_{:.4}.csv'.format(learning_rate, drop_prob, alpha), 'a', encoding='utf-8') as f:
                        writer = csv.writer(f)
                        # writer.writerow([loss_val, accuracy, precision, recall, f1, mcc])
                        writer.writerow([loss_val, accuracy, closeness])
                    f.close()
                    # print("dev loss: ", loss_val, "dev accuracy: ", accuracy, "precision: ", precision, "recall: ", recall, "f1: ", f1, "mcc: ", mcc)
                    print("dev loss: ", loss_val, "dev accuracy: ", accuracy, "closeness: ", closeness)
                checkpoint += 1

    else:
        # testing case
        print("testing data, loading from path" + save_dir + " ...")
        model = torch.load(save_dir)
        loss_val, accuracy, closeness = evaluate(model, test_iterator, device)
        print("dev loss: ", loss_val, "dev accuracy: ", accuracy, "closeness: ", closeness)


if __name__=="__main__":
    main()
