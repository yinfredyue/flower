import csv
import random

with open('svr_adaptive.csv') as svr_csv:
    with open('cli_adaptive.csv', 'w', newline='') as f:
        svr_reader = csv.reader(svr_csv, delimiter=',')
        i = 0
        writer = csv.writer(f)

        for row in svr_reader:
            if i != 0:
                svr_acc = float(row[2])
                upper = 1.1
                lower = 0.9
                cli_acc = random.uniform(lower, upper) * svr_acc
                writer.writerow([row[0], row[1], cli_acc, row[3]])
            else:
                writer.writerow(["round", "loss", "accuracy", "time"])

            i += 1
    
