import processing as prc

from sklearn.metrics import mean_squared_error

from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR

def forecast_mlp(X, y, solver, hidden_neurons, learning_rate, activation, jumps):

    lst_results =[]

    X_train, X_valid, X_test, y_train, y_valid, y_test = prc.split_dataset(X, y)

    X_train_norm, X_valid_norm, X_test_norm = prc.norm_X_dataset(X_train, X_valid, X_test)

    y_train_norm, y_valid_norm = prc.norm_y_dataset(y_train, y_valid)

    for i in range(jumps):
        sd=i

        best_error = float('inf')
        best_rna = None

        for h in hidden_neurons:
            for l in learning_rate:
                for a in activation:
                    #[NEURON ESCONDIDOS, TAXA DE APRENDIZADO, FUNCAO DE ATIVACAO]
                    rna = MLPRegressor(hidden_layer_sizes=(h,),learning_rate_init=l,activation=a,shuffle=False, random_state=sd, solver=solver)
                    
                    rna.fit(X_train_norm,y_train_norm) # Treina o modelo com os dados de treinamento

                    preds = rna.predict(X_valid_norm)
                    error = mean_squared_error(y_valid_norm, preds) 

                    if error < best_error:
                        best_rna = rna
                        best_error = error

        pred_test = best_rna.predict(X_test_norm)
        pred_test_denom = prc.denorm_data(y_train, pred_test)
        error_test = mean_squared_error(y_test, pred_test_denom)
        lst_results.append(error_test)
    
    return lst_results, pred_test, pred_test_denom, X_test, y_test, best_rna

def forecast_svr(X, y, C_values, epsilon_values, gamma_values):

    lst_results =[]

    X_train, X_valid, X_test, y_train, y_valid, y_test = prc.split_dataset(X, y)

    X_train_norm, X_valid_norm, X_test_norm = prc.norm_X_dataset(X_train, X_valid, X_test)

    # SVR é sensível à escala do X, mas não precisa normalizar y
    # y_train_norm, y_valid_norm = prc.norm_y_dataset(y_train, y_valid)

    best_error = float('inf')
    best_svr = None

    for c in C_values:
        for e in epsilon_values:
            for g in gamma_values:

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

