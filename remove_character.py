import csv
import random

def split_data(input_file):
    with open(input_file, 'r') as f:
        reader = csv.reader(f)
        rows = list(reader)
        num_total = len(rows)
        num_train = int(0.8 * num_total)
        num_valid = int(0.1 * num_total)
        num_test = num_total - num_train - num_valid
        train = rows[:num_train]
        valid = rows[num_train:num_train + num_valid]
        test = rows[:-num_test]
    f.close()

    with open('train_data.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        for sample in train:
            writer.writerow(sample)
    f.close()

    with open('valid_data.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        for sample in valid:
            writer.writerow(sample)
    f.close()

    with open('test_data.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        for sample in test:
            writer.writerow(sample)
    f.close()

def main():
    with open('removed_characters_buckets_sentiments.csv', 'r') as in_file:
        with open ('fixed_lines.csv', 'w') as out_file:
            data = in_file.readlines()
            random.shuffle(data)
            out_data = []
            for line in data:
                if line:
                    out_data.append(line)
            rows = '\n'.join([row.strip() for row in out_data])
            out_file.write(rows)
    in_file.close()
    out_file.close()
    return

if __name__=="__main__":
    main()
