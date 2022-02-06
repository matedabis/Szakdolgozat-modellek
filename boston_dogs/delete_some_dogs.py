import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import cv2
import re
import matplotlib.pyplot as plt
import os
import csv

y= pd.read_csv('../../labels.csv')
y.head()

train_file_location = '../../train/'
train_data = y.assign(img_path = lambda x : train_file_location + x['id'] + '.jpg')
train_data.head()

for i in range(len(y["breed"])):
    if (y["breed"][i] not in ["golden_retriever", "boxer", "doberman"]):
        if os.path.exists(train_data["img_path"][i]):
            os.remove(train_data["img_path"][i])


with open('../../labels.csv', 'r') as inp, open('../../labels_new.csv', 'w') as out:
    writer = csv.writer(out)
    writer.writerow(["id", "breed"])
    for row in (csv.reader(inp)):
        if (row[1] in ["golden_retriever", "boxer", "doberman"]):
            writer.writerow(row)

f=pd.read_csv("../../sample_submission.csv")
keep_col = ["id", "golden_retriever", "boxer", "doberman"]
new_f = f[keep_col]
new_f.to_csv("../../sample_submission_new.csv", index=False)
