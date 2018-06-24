import pandas as pd
import numpy as np
from datetime import datetime
import os
import argparse

CSV_DIR = 'csv'
os.makedirs(CSV_DIR, exist_ok=True)

parser = argparse.ArgumentParser()
parser.add_argument('-c', '--csvs',    nargs='+', default=[], help='Paths to CSVs')
parser.add_argument('-o', '--output',  default=None, help='Paths to CSVs')
a = parser.parse_args()

a.output = a.output or '%s/geoensemble_%s.csv' % (CSV_DIR, datetime.now().strftime("%Y-%m-%d-%H%M"))

test_predicts_list = []
for name in a.csvs:
    b1 = pd.read_csv(name)
    test_predicts_list.append(b1['deal_probability'].values)

test_predicts = np.ones(test_predicts_list[0].shape)
for fold_predict in test_predicts_list:
    test_predicts *= fold_predict

test_predicts **= (1. / len(test_predicts_list))  


subm = pd.read_csv('sample_submission.csv', usecols=['item_id'])
subm['deal_probability'] = np.clip(test_predicts, 0, 1)
subm.to_csv(a.output, index=False)
