import numpy as np

import processing as prc

from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor


def forecast_rf(X, y, n_estimators_values, max_depth_values):

    lst_results =[]

    X_train, X_valid, X_test, y_train, y_valid, y_test = prc.split_dataset(X, y)

    X_train_norm, X_valid_norm, X_test_norm = prc.norm_X_dataset(X_train, X_valid, X_test)

    # RF não é sensível à escala — normalização opcional, mas mantém consistência
    # y_train_norm, y_valid_norm = prc.norm_y_dataset(y_train, y_valid)

    best_error = float('inf')
    best_rf = None

    for n in n_estimators_values:
        for d in max_depth_values:

            rf = RandomForestRegressor(n_estimators=n, max_depth=d, random_state=42)
            rf.fit(X_train_norm, y_train)

            preds = rf.predict(X_valid_norm)
            error = mean_squared_error(y_valid, preds)

            if error < best_error:
                best_rf = rf
                best_error = error

    pred_test = best_rf.predict(X_test_norm)
    error_test = mean_squared_error(y_test, pred_test)
    lst_results.append(error_test)
    
    return lst_results, pred_test, X_test, y_test, best_rf