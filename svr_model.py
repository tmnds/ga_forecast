import numpy as np

import processing as prc

from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error


def forecast_svr(X, y, C_values, epsilon_values, gamma_values):

    lst_results = []

    X_train, X_valid, X_test, y_train, y_valid, y_test = prc.split_dataset(X, y)

    X_train_norm, X_valid_norm, X_test_norm = prc.norm_X_dataset(X_train, X_valid, X_test)

    # SVR é sensível à escala do X, mas não precisa normalizar y
    # y_train_norm, y_valid_norm = prc.norm_y_dataset(y_train, y_valid)

    best_error = float('inf')
    best_svr = None

    for c in C_values: # variáveis de folga
        for e in epsilon_values: # largura do hipertubo
            for g in gamma_values: # parâmetro de kernel

                svr = SVR(kernel='rbf', C=c, epsilon=e, gamma=g)
                svr.fit(X_train_norm, y_train) # Treina o modelo com os dados de treinamento
                
                preds = svr.predict(X_valid_norm)
                error = mean_squared_error(y_valid, preds) 

                if error < best_error:
                    best_svr = svr
                    best_error = error

    pred_test = best_svr.predict(X_test_norm)
    error_test = mean_squared_error(y_test, pred_test)
    lst_results.append(error_test)
    
    return lst_results, pred_test, X_test, y_test, best_svr