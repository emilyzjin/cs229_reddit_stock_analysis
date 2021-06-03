import torch
import torch.nn as nn
import torchtext
import csv
from util import get_available_devices
from sentiment_util import evaluate, data_preprocess
from models.sentiment_model import WithoutSentiment, WithSentiment, SentimentLSTM
from torchtext.legacy import data
import torch.optim as optim


def tokenize(s):
    return s.split(' ')


TEXT = data.Field(tokenize=tokenize, lower=True, include_lengths=True)
UPVOTE = data.LabelField(sequential=False, use_vocab=False, dtype=torch.int64)
CHANGE = data.LabelField(sequential=False, use_vocab=False, dtype=torch.float)
SENT = data.LabelField(sequential=False, use_vocab=False, dtype=torch.int64)
LABEL = data.LabelField(sequential=False, use_vocab=False, dtype=torch.int64)

def main():
    # create_csv()
    train = True
    batch_size = 2048
    hidden_size = 256
    output_dim = 1
    drop_prob = 0.5
    learning_rate = 1e-3 # TODO: hyper
    num_epochs = 100
    beta1, beta2 = 0.9, 0.999 # for Adam
    alpha = 1.0 # for ELU # TODO: hyper
    max_grad_norm = 2.0
    print_every = 100
    use_sentiment = True
    save_dir = 'results/model.path_lr_{:.4}_drop_prob_{:.4}_alpha_{:.4}.tar'.format(learning_rate, drop_prob, alpha)

    device, gpu_ids = get_available_devices()

    train_iterator, valid_iterator, test_iterator = data_preprocess(TEXT, UPVOTE, CHANGE, SENT, LABEL, data, 30000, device, batch_size)

    # Initialize model.
    if use_sentiment:
        sent_model = SentimentLSTM(
                        vocab_size = len(TEXT.vocab),
                        embedding_dim = 100,
                        hidden_dim=hidden_size,
                        output_dim=output_dim,
                        n_layers=2,
                        bidirectional=True,
                        dropout=drop_prob,
                        pad_idx=TEXT.vocab.stoi[TEXT.pad_token],
                        device=device
        )
        sent_model.load_state_dict(torch.load('trained_sentiment.pt', map_location=torch.device(device)))
        model = WithSentiment(
            hidden_dim=hidden_size,
            alpha=alpha
        )
    else:
        sent_model = None
        model = WithoutSentiment(
            hidden_dim=hidden_size,
            alpha=alpha
        )
    sent_model = sent_model.to(device)
    model = model.to(device)
    # Initialize optimizer
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, betas=(beta1, beta2))

    # Training Loop
    if train:
        checkpoint = 0
        for epoch in range(num_epochs):
            iter = 0
            with torch.enable_grad():
                for batch in train_iterator:
                    optimizer.zero_grad()
                    # Grab labels
                    target = batch.label
                    # Grab multimodal data
                    if use_sentiment:
                        text, text_lengths = batch.text
                        text, text_lengths = text.to(device), text_lengths.to(device)
                        sent = sent_model(text, text_lengths)
                        multimodal_data = torch.cat((batch.upvote.unsqueeze(dim=1), # upvotes
                                                     batch.change.unsqueeze(dim=1), # past week change
                                                     sent), # sentiments
                                                     dim=1)
                    else:
                        multimodal_data = torch.cat((batch.upvote.unsqueeze(dim=1), # upvotes
                                                     batch.change.unsqueeze(dim=1)), # past week change
                                                     dim=1)
                        sent_model = None
                    # Apply model
                    y = model(batch, multimodal_data)
                    target = target.to(device)
                    loss_function = nn.CrossEntropyLoss()
                    loss = loss_function(y, target)
                    loss_val = loss.item()

                    # Backward
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                    optimizer.step()

                    if iter % print_every == 0:
                        print('Epoch:{}, Iter: {}, Loss:{:.4}'.format(epoch, iter, loss.item()))
                    iter += 1

                # if checkpoint % 3 == 0:
                    print("evaluating on dev split...")
                    loss_val, accuracy, precision, closeness, recall, f1, mcc = evaluate(model, valid_iterator, device, use_sentiment, sent_model)
                    # loss_val, accuracy, closeness = evaluate(model, valid_iterator, device)
                    with open('results/model.path_lr_{:.4}_drop_prob_{:.4}_alpha_{:.4}.csv'.format(learning_rate, drop_prob, alpha), 'a', encoding='utf-8') as f:
                        writer = csv.writer(f)
                        writer.writerow([loss_val, accuracy, closeness, precision, recall, f1, mcc])
                        # writer.writerow([loss_val, accuracy, closeness])
                    f.close()
                    print("dev loss: ", loss_val, "dev accuracy: ", accuracy, "closeness: ", closeness)
                    print("precision: ", precision)
                    print("recall: ", recall)
                    print("f1: ", f1)
                    print("mcc: ", mcc)
                    # print("dev loss: ", loss_val, "dev accuracy: ", accuracy, "closeness: ", closeness)
                checkpoint += 1

                torch.save(model, save_dir)

    else:
        # testing case
        print("testing data, loading from path" + save_dir + " ...")
        model = torch.load(save_dir)
        loss_val, accuracy, closeness = evaluate(model, test_iterator, device)
        print("dev loss: ", loss_val, "dev accuracy: ", accuracy, "closeness: ", closeness)


if __name__=="__main__":
    main()
