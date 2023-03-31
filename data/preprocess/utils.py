import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')


def check_nan(data):
    """
    Prints the number of NaN values in each column of a pandas DataFrame.

    Parameters:
    -----------
    data : pandas.DataFrame
        The DataFrame to check for NaN values.

    Returns:
    --------
    None
    """

    for col in data.columns:
        num_nan = data[col].isna().sum()
        print(f'Number of NaN values in column {col}: {num_nan}')


def count_unique(data, columns):
    """
    Prints the number of unique values in each column of a pandas DataFrame.

    Parameters:
    -----------
    data : pandas.DataFrame
        The DataFrame to check for unique values.
    columns : list of str
        The columns to count the unique values for.

    Returns:
    --------
    None
    """

    for col in columns:
        num_unique = data[col].nunique()
        print(f'Number of unique values in {col}: {num_unique}')


def remove_log(data, user_ind, num=2):
    """
    Removes users from a DataFrame based on the number of times they appear in the `user_ind` column.

    Parameters:
    -----------
    data : pandas.DataFrame
        The DataFrame to remove users from.
    user_ind : str
        The column containing the user IDs.
    num : int, optional
        The minimum number of times a user should appear in `user_ind` to be kept in the DataFrame.
        Default is 2.

    Returns:
    --------
    pandas.DataFrame
        The DataFrame with the specified users removed.
    """

    user_counts = data[user_ind].value_counts()
    users_to_remove = user_counts[user_counts < num].index
    data = data[~data[user_ind].isin(users_to_remove)]
    return data

