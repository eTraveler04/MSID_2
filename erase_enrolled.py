#!/usr/bin/env python3
import csv
import argparse

def remove_rows_with_target_1(infile: str, outfile: str):
    with open(infile, newline='', encoding='utf-8') as fin, \
         open(outfile, 'w', newline='', encoding='utf-8') as fout:
        reader = csv.DictReader(fin)
        writer = csv.DictWriter(fout, fieldnames=reader.fieldnames)
        writer.writeheader()
        for row in reader:
            # pomijamy wiersz, gdy Target == '1'
            if row.get('Target') != '1':
                writer.writerow(row)

if __name__ == '__main__':
    p = argparse.ArgumentParser(description="Usuń wiersze z Target == 1")
    p.add_argument('input_csv',  help="ścieżka do oryginalnego pliku CSV")
    p.add_argument('output_csv', help="ścieżka do przefiltrowanego pliku CSV")
    args = p.parse_args()
    remove_rows_with_target_1(args.input_csv, args.output_csv)


