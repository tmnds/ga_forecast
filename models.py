import processing as prc

from sklearn.metrics import mean_squared_error

from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR

class MLP():

    def __init__(self, input_windows, target_values):

        self.input_windows = input_windows
        self.target_values = target_values

        self.hidden_neurons = [10,20,30,40]
        self.learning_rate = [0.1, 0.01, 0.001]
        self.activation = ['logistic','tanh','relu']
        self.solver = 'sgd'
        
        self.lst_results = []
        self.best_errors_list = []
        self.best_error = float('inf')
        self.best_rna = None
    
    def normalize_data(self):

        # Split Temporal os dados
        train_window, valid_window, test_window, train_target, valid_target, test_target = prc.split_dataset(self.input_windows, self.target_values)

        # Normalização das janelas
        train_window_norm, valid_window_norm, test_window_norm = prc.norm_X_dataset(train_window, valid_window, test_window)
       
        # Normalização de Alvos
        train_target_norm, valid_target_norm = prc.norm_y_dataset(train_target, valid_target)
        
        return {
            'train_window_norm': train_window_norm,
            'valid_window_norm': valid_window_norm,
            'test_window_norm': test_window_norm,
            'train_target_norm': train_target_norm,
            'valid_target_norm': valid_target_norm,
            
            'test_window': test_window,
            'train_target': train_target,
            'test_target': test_target
        }
        
    def get_best_error(self, rna, error):

        if error < self.best_error:
            self.best_rna = rna
            self.best_error = error
            self.best_errors_list.append({'erro': error, 'params': self.best_rna.get_params()})
        
        # return self.best_rna, self.best_error
    
    def get_predict(self, normalized):

        pred_test = self.best_rna.predict(normalized['test_window_norm'])
        pred_test_denom = prc.denorm_data(normalized['train_target'], pred_test)
        error_test = mean_squared_error(normalized['test_target'], pred_test_denom)
        self.lst_results.append(error_test)

        return {
            'pred_test': pred_test,
            'pred_test_denom': pred_test_denom,
            'error_test': error_test
        }

    def grid_search(self, sd, normalized):
        
        h = []

        for h in self.hidden_neurons:
            for l in self.learning_rate:
                for a in self.activation:
                    rna = MLPRegressor(hidden_layer_sizes=(h,),learning_rate_init=l,activation=a,shuffle=False, random_state=sd, solver=self.solver)
                    
                    rna.fit(normalized['train_window_norm'],normalized['train_target_norm']) # Treina o modelo com os dados de treinamento

                    preds = rna.predict(normalized['valid_window_norm'])
                    error = mean_squared_error(normalized['valid_target_norm'], preds) 

                    rna, error = self.get_best_error(rna, error)
        
        return rna, error
    
    def train_model(self):

        jumps = 10
        normalized = self.normalize_data()

        for i in range(jumps):
            sd=i

            self.grid_search(sd, normalized)   
            predict = self.get_predict(normalized) 
        
        return {
            
            'lst_results': self.lst_results,
            'pred_test': predict['pred_test'],
            'pred_test_denom': predict['pred_test_denom'],
            'test_window': normalized['test_window'],
            'test_target': normalized['test_target'],
            'best_rna': self.best_rna,
            'best_errors_list': self.best_errors_list
        }


def forecast_mlp(X, y, solver, hidden_neurons, learning_rate, activation, jumps):

    lst_results =[]

    # Divisão do dataset e Normalizaçao
    X_train, X_valid, X_test, y_train, y_valid, y_test = prc.split_dataset(X, y)

    X_train_norm, X_valid_norm, X_test_norm = prc.norm_X_dataset(X_train, X_valid, X_test)
    y_train_norm, y_valid_norm = prc.norm_y_dataset(y_train, y_valid)

    best_errors_list = []

    best_error = float('inf')
    best_rna = None

    for i in range(jumps):
        sd=i

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
                        best_errors_list.append({'erro': error,'params': best_rna.get_params()})

        pred_test = best_rna.predict(X_test_norm)
        pred_test_denom = prc.denorm_data(y_train, pred_test)
        error_test = mean_squared_error(y_test, pred_test_denom)
        lst_results.append(error_test)
    
    return lst_results, pred_test, pred_test_denom, X_test, y_test, best_rna, best_errors_list

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

