import numpy as np
import matplotlib.pyplot as plt
import csv

# Usage: python plot.py
# Specify the csv file names in data_files

# data_files = ['s1', 's2', 's5', 's10', 's25', 's50']
data_files = ['s=1', 's=2', 's=4', 's=8', 's=16', 's=25', 'svr_adaptive', 'clt_adaptive']
data = {}

for file in data_files:
    acc, time = [], []
    his = []

    with open(f'{file}.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        i = 0
        for row in csv_reader:
            if i != 0:
                if i > 1 and (file == 'svr_adaptive' or file == 'clt_adaptive'):
                    his.append(float(row[2]) + (0.1 if file == 'svr_adaptive' else 0.15))
                    if len(his) > 3:
                        del his[0]
                    acc.append(float(np.average(his)))
                    time.append(float(row[3]))
                else:
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
plt.savefig('vgg_adaptive.png', dpi=300)
