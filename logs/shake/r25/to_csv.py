import csv
import argparse
import sys
from typing import List
import re

# Usage:
# python to_csv.py --log s1.log --patterns progress --output s1.csv

parser = argparse.ArgumentParser("convert .log to .csv")
parser.add_argument("--log", type=str, required=True)
parser.add_argument("--patterns", nargs='+', required=True)
parser.add_argument("--output", type=str, required=True)
args = parser.parse_args()

# Perform a "grep" in python
def grep(file_name, pattern: List[str]):
    file = open(file_name, "r")
    res = []

    for line in file:
        for p in pattern:
            if p in line:
                res.append(line)
                break
    
    return res


def main():
    lines = grep(args.log, args.patterns)

    with open(args.output, 'w', newline='') as f:
        writer = csv.writer(f)

        writer.writerow(["round", "loss", "accuracy", "time"])

        for line in lines:
            # Parse number: https://stackoverflow.com/a/29581287/9057530
            r = re.findall("[-+]?[.]?[\d]+(?:,\d\d\d)*[\.]?\d*(?:[eE][-+]?\d+)?", line)

            # Extra (round, loss, accuracy, time)
            # ['2021', '-04', '-05', '02', '39', '34,048', '173', '1', '24', '1742.0151271820068', '0.2222222222222222', '2451.7025870867074']
            r = r[8:]

            # Write to csv
            # https://www.programiz.com/python-programming/writing-csv-files
            writer.writerow(r)


if __name__ == "__main__":
    main()