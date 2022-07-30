import os
import sys
import gzip
import time
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

def check_nan(data):
    for index in data.columns:
        num_nan = data[data[index].isna()].shape[0]
        print('number of NaN in column '+index+':', num_nan)

def count_unique(data, columns):
    for index in columns:
        print('number of unique value in '+index+':', data[index].unique().shape[0])

def remove_log(data, user_ind, num=2):
    user_count = data[user_ind].value_counts() 
    user_count = user_count[user_count<num] 
    user_remove_ind = list(user_count.index)
    data = data[~data[user_ind].isin(user_remove_ind)] 
    return data