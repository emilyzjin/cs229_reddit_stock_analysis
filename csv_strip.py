import csv

with open('reddit_dataset.csv') as in_file:
    with open('removed_characters.csv', 'w') as out_file:
        writer = csv.writer(out_file)
        for row in csv.reader(in_file):
            if row:
                row = [val.replace('\'', '').replace('\"', '').replace('[', '').replace(']', '') for val in row]
                writer.writerow(row)