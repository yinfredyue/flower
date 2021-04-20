import numpy as np
import matplotlib.pyplot as plt
import csv

# Usage: python plot.py
# Specify the csv file names in data_files

# data_files = ['s1', 's2', 's5', 's10', 's25', 's50']
data_files = ['s=1', 's=2', 's=5', 's=10', 's=25', 'svr_adaptive', 'cli_adaptive']
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
    if series == 'svr_adaptive':
        plt.plot(time, acc, 'o-', label=series, color='#e377c2')
    elif series == 'cli_adaptive':
        plt.plot(time, acc, '^-', label=series, color='#8c564b')
    else:
        plt.plot(time, acc, '-.', label=series)

plt.xlabel("Time (sec)")
plt.ylabel("Accuracy")
plt.legend(loc='lower right')
plt.savefig('lenet_adaptive.png', dpi=300)
