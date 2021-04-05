import numpy as np
import matplotlib.pyplot as plt
import csv

# Usage: python plot.py
# Specify the csv file names in data_files

data_files = ['s1', 's2', 's4', 's8', 's16']
data = {}

for file in data_files:
    acc, time = [], []

    with open(f'{file}.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        i = 0
        for row in csv_reader:
            if i != 0:
                acc.append(float(row[2]))
                time.append(float(row[3]))

            i += 1
    
    data[file] = np.array(acc), np.array(time)

for series in data:
    acc, time = data[series]
    plt.plot(time, acc, label=series)

plt.xlabel("Time (sec)")
plt.ylabel("Accuracy")
plt.legend(loc='lower right')
plt.savefig('stale.png', dpi=300)
