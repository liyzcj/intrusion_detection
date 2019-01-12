import pandas as pd
import numpy as np


def load_data():
    """
    Return (x_train, y_train), (x_test, y_test).
    """
    # load data from file
    x_train = pd.read_csv("data/x_train.csv")
    y_train = pd.read_csv("data/y_train.csv")
    x_test = pd.read_csv("data/x_test.csv")
    y_test = pd.read_csv("data/y_test.csv")
    
    # data normalization
    mean = x_train.mean()
    std = x_train.std()
    x_train = (x_train - mean) / std
    x_test = (x_test - mean) / std
    
    # convert dataframe to numpy array
    x_train = x_train.values
    y_train = y_train.values
    x_test = x_test.values
    y_test = y_test.values
    
    return (x_train, y_train),(x_test, y_test)