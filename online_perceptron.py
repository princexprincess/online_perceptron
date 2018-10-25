#!/usr/bin/env python3
import csv
import numpy as np
import pandas as pd
csv_tr = csv.reader(open('reviews_tr.csv','r'))
csv_te = csv.reader(open('reviews_te.csv','r'))
data_header = next(csv_tr)
data = []
for row in csv_tr:
    row[1] = row[1].split()
    data.append(row)   
print('done')
print(len(data))
print(data[0])
