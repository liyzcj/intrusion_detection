import pandas as pd
import numpy as np


def load_data(norm_mode="zero_mean"):
    """
    Return (x_train, y_train), (x_test, y_test).
    """
    # load data from file
    x_train = pd.read_csv("data/x_train.csv")
    y_train = pd.read_csv("data/y_train.csv", header=None)
    x_test = pd.read_csv("data/x_test.csv")
    y_test = pd.read_csv("data/y_test.csv", header=None)
    
    # data normalization
    if norm_mode=="zero_mean":
        mean = x_train.mean()
        std = x_train.std()
        x_train = (x_train - mean) / std
        x_test = (x_test - mean) / std
    if norm_mode=="minmax":
        maxi = x_train.max()
        mini = x_train.min()
        x_train = (x_train - mini) / (maxi - mini)
        x_test = (x_test - mini) / (maxi - mini)
    # convert dataframe to numpy array
    x_train = x_train.values
    y_train = y_train.values
    x_test = x_test.values
    y_test = y_test.values
    
    return (x_train, y_train),(x_test, y_test)