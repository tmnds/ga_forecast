import numpy as np
import pandas as pd

def window_lags(data, n_lags):
    X, y = [], []
    for i in range(n_lags, len(data)):
        X.append(data[i - n_lags:i])
        y.append(data[i])

    return np.array(X), np.array(y)
    # return pd.DataFrame(X), pd.Series(y)

def split_dataset(X, y, train_size=0.6, valid_size=0.2):

    split = int(len(X) * train_size)
    split_valid = int(len(X) * valid_size)

    X_train, X_valid, X_test = X[:split], X[split:split+split_valid], X[split+split_valid:]
    y_train, y_valid, y_test = y[:split], y[split:split+split_valid], y[split+split_valid:]

    return X_train, X_valid, X_test, y_train, y_valid, y_test

def norm_X_dataset(X_train, X_valid, X_test):
    # Normalização - Evitando data leakage
    # Xn = min + (X - Xmin) / (Xmax - Xmin) . (max - min)
    X_min = X_train.min()
    X_max = X_train.max()

    X_train_norm = (X_train - X_min) / (X_max - X_min)
    X_valid_norm = (X_valid - X_min) / (X_max - X_min)
    X_test_norm = (X_test - X_min) / (X_max - X_min)

    return X_train_norm, X_valid_norm, X_test_norm

def norm_y_dataset(y_train, y_valid):
    # Normalização - Evitando data leakage
    # yn = min + (y - ymin) / (ymax - ymin) . (max - min)
    y_min = y_train.min()
    y_max = y_train.max()

    y_train_norm = (y_train - y_min) / (y_max - y_min)
    y_valid_norm = (y_valid - y_min) / (y_max - y_min)

    return y_train_norm, y_valid_norm

def denorm_data(y_train, data):
    y_min = y_train.min()
    y_max = y_train.max()

    return data * (y_max - y_min) + y_min