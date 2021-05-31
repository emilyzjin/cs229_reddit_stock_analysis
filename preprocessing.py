import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda
import matplotlib.pyplot as plt 
import yfinance as yf
import csv
from googlesearch import search

"""
***Required dowloads***
(in addition to those already in the cs229 conda env):

pip install google
pip install yfinance
conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch
pip install pytorch-transformers
"""

day_indices = {1: 5, 2: 6, 3: 7, 7: 8}

name_ticker_dict = {'Yahoo': 'AABA'}

def get_data(csv_file):
    """
    USE THIS FUNCTION to get the data to a nice format use:
    from preprocessing import get_data
    returns a list of lists, each row represents [tweet, ticker, 7 day percent change]
    """
    with open(csv_file, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        result = list(reader)
    return result

def read_all_data(csv_file): 
    """
    Data format (15 long): 
    ID number, 0 
    TWEET, 1
    STOCK, 2
    DATE, 3
    LAST_PRICE, 4
    1_DAY_RETURN, 5
    2_DAY_RETURN, 6
    3_DAY_RETURN, 7
    7_DAY_RETURN, 8
    'PX_VOLUME', 9
    'VOLATILITY_10D', 10
    'VOLATILITY_30D', 11
    'LSTM_POLARITY', 12 
    'TEXTBLOB_POLARITY', 13
    'MENTION', 14 
    """

    with open(csv_file, "r", encoding="utf8") as f:
        reader = csv.reader(f, delimiter = ",")
        data = list(reader)
    data = data[1:]

    # clean up data
    length = len(data)
    counter = 0
    for i in reversed(range(length)):
        if data[i] == []:
            del data[i]
        elif len(data[i]) == 1:
            data[i-1][-1] += data[i][0]
            del data[i]
        elif len(data[i]) == 2: 
            data[i].extend(data[i+1][1:])
            del data[i+1]
    
    # turn all comapany names to tickers

    for i, row in enumerate(data):
        if row[2] not in name_ticker_dict.keys():
            row[2] = name_to_ticker(row[2])
        else: 
            row[2] = name_ticker_dict[row[2]]
        if (i % 100 == 99 and i < 2000) or (i % 1000 == 999 and i < 10000) or (i % 10000 == 9999):
            j = (i + 1) / len(data) * 100
            print("%i rows completed, %i percent done" % (i + 1, j))

    # split to train, dev, test

    train_data = data[:-10000]
    dev_data = data[-10000:-5000]
    test_data = data[-5000:]
    
    print("train data size: %i, dev data size: %i, test data size: %i" 
    % (len(train_data), len(dev_data), len(test_data)))

    return train_data, dev_data, test_data

def read_days_data(csv_file, days):
    """
    Data format (3 long):
    Tweet, 0
    stock, 1
    7 day percent change, 2
    """
    all_train, all_dev, all_test = read_all_data(csv_file)
    d_col = day_indices[days]
    return [[row[1], row[2], row[d_col]] for row in all_train], [[row[1], row[2], row[d_col]] for row in all_dev], [[row[1], row[2], row[d_col]] for row in all_test] 

def name_to_ticker(name):
    """
    Searches a the keyword "yahoo finance stock_name" on google, 
    pulls the ticker from the yahoo finance page. 
    Credit: 
    This function is based off of https://github.com/MakonnenMak/company-name-to-ticker-yahoo-finance.git
    """
    searchval = 'yahoo finance ' + name
    link = []
    for url in search(searchval, tld='es', lang='es', stop=1):
        link.append(url)
    link = str(link[0])
    link=link.split("/")
    if link[-1]=='':
        ticker=link[-2]
    else:
        x=link[-1].split('=')
        ticker=x[-1]
    name_ticker_dict[name] = ticker
    return ticker

def write_name_ticker_dict_csv():
    with open('name_ticker_dict.csv', 'w') as f:  
        writer = csv.writer(f)
        for key, value in name_ticker_dict.items():
            writer.writerow([key, value])
    print("name_ticker_dict.csv file written")
    return

def write_csv(data, csv_file):
    with open(csv_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerows(data)
    print("CSV file %s written" % (csv_file))
    return

def main():
    train, dev, test = read_days_data("stockreturnpred/Dataset-release version/reduced_dataset-release.csv", 7)
    write_name_ticker_dict_csv()
    write_csv(train, "train.csv")
    write_csv(dev, "dev.csv")
    write_csv(test, "test.csv")
    
    
if __name__=="__main__":
    main()