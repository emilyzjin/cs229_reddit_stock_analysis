import numpy as np
from preprocessing import get_data
from util import create_buckets, get_tweets_change
import torch
import torch.nn as nn

results_files = ['results_learn_rate=0.1', 'results_learn_rate=0.01',
                 'results_learn_rate=0.001', 'results_learn_rate=0.0001']
_, true_labels = get_tweets_change('test.csv')
true_labels = true_labels[:, np.newaxis] # reshape to nx1


def evaluate(results_file):
    y_hat = np.asarray(get_data(results_file+'.csv'), dtype=float)
    y = true_labels

    bucket_y_hat = create_buckets(y_hat)
    bucket_y = create_buckets(y)

    precisions, recalls, f1s, mccs = calc_one_v_many_metrics(bucket_y_hat, bucket_y)

    print(f'Precision: {precisions}')
    print(f'Recalls: {recalls}')
    print(f'F1s: {f1s}')
    print(f'MCCS: {mccs}')


def calc_one_v_many_metrics(y_hat, y):
    precisions = []
    recalls = []
    f1s = []
    mccs = []
    for i in range(5):
        y_hat_binary = torch.where(y_hat == i, 1, 0)
        y_binary = torch.where(y == i, 1, 0)
        precisions.append(calc_precision(y_hat_binary, y_binary))
        recalls.append(calc_recall(y_hat_binary, y_binary))
        f1s.append(calc_f1(y_hat_binary, y_binary))
        mccs.append(calc_mcc(y_hat_binary, y_binary))
    return precisions, recalls, f1s, mccs


def calc_precision(y_hat_binary, y_binary):
    tp, fp, tn, fn = calc_numbers(y_hat_binary, y_binary)
    precision = (tp + 1) / (tp + fp + 1)
    return precision


def calc_recall(y_hat_binary, y_binary):
    tp, fp, tn, fn = calc_numbers(y_hat_binary, y_binary)
    recall = (tp + 1) / (tp + fn + 1)
    return recall


def calc_f1(y_hat_binary, y_binary):
    precision = calc_precision(y_hat_binary, y_binary)
    recall = calc_recall(y_hat_binary, y_binary)
    f1 = (2 * precision * recall + 1) / (precision + recall + 1)
    return f1


def calc_mcc(y_hat_binary, y_binary):
    tp, fp, tn, fn = calc_numbers(y_hat_binary, y_binary)
    mcc = (tp * tn - fp * fn + 1) / torch.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn) + 1)
    return mcc


def calc_numbers(preds_binary, labels_binary):
    tp = torch.sum(torch.logical_and(preds_binary, labels_binary))
    fp = torch.sum(torch.logical_and(preds_binary, labels_binary == torch.zeros_like(labels_binary)))
    tn = torch.sum(torch.logical_and(preds_binary == torch.zeros_like(preds_binary), labels_binary == torch.zeros_like(labels_binary)))
    fn = torch.sum(torch.logical_and(preds_binary == torch.zeros_like(preds_binary), labels_binary == 1))

    return tp, fp, tn, fn


def macroaverage(metrics):
    return np.mean(metrics)


# def microaverage(metrics):


def calc_accuracy(y_hat, y):
    return np.mean(y_hat == y)


def main():
    for file in results_files:
        print(f'Evaluation Results for {file}.csv')
        evaluate(file)
        print('----------------')


if __name__ == "__main__":
    main()
