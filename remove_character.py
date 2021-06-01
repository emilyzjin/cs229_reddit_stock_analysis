import csv
import random

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
