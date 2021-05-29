import csv

def convert_data():
    with open("all_tickers_list.csv", "r", encoding="utf8") as f:
        reader = csv.reader(f)
        t = list(reader)
    ticker_list = [x[0].lower() for x in t[1:]]
    with open("tickers_dict.csv", "r", encoding="utf8") as f:
        reader = csv.reader(f)
        ticker_dict = {rows[0]:rows[1] for rows in reader}
    return ticker_list, ticker_dict

def main():
    ticker_list, ticker_dict = convert_data()
    print(ticker_list[:10], ticker_dict)
    
    
if __name__=="__main__":
    main()