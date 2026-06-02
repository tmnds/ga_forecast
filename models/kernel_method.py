import processing as prc

from sklearn.metrics import mean_squared_error
from sklearn.svm import SVR

class ForecastSVM:
    def __init__(self, input_windows, target_values):

        self.input_windows = input_windows
        self.target_values = target_values
        
        self.C_values=[0.1, 1, 10, 100] # Penalidade por erro
        self.epsilon_values=[0.01, 0.05, 0.1]
        self.gamma_values=['scale', 0.01, 0.1]

        self.lst_results = []
        self.best_errors_list = []
        self.best_error = float('inf')
        self.best_svr = None

    def normalize_data(self):
        input_train, input_valid, input_test, target_train, target_valid, target_test = prc.split_dataset(self.input_windows, self.target_values)

        input_train_norm, input_valid_norm, input_test_norm = prc.norm_X_dataset(input_train, input_valid, input_test)

        # SVR é sensível à escala do X, mas não precisa normalizar y
        # y_train_norm, y_valid_norm = prc.norm_y_dataset(y_train, y_valid)
        return {
            'input_train_norm': input_train_norm,
            'input_valid_norm': input_valid_norm,
            'input_test_norm': input_test_norm,
            'target_train': target_train,
            'target_valid': target_valid,
            'target_test': target_test
        }
    
    def update_best_model(self, svr, error):
        
        if error < self.best_error:

            self.best_svr = svr
            self.best_error = error
            
            self.best_errors_list.append({'erro': error, 'params': self.best_svr.get_params()})

    def grid_search(self, normalized):

        for c in self.C_values:
            for e in self.epsilon_values:
                for g in self.gamma_values:

                    svr = SVR(kernel='rbf', C=c, epsilon=e, gamma=g)
                    svr.fit(normalized['input_train_norm'], normalized['target_train'])
                
                    preds = svr.predict(normalized['input_valid_norm'])
                    error = mean_squared_error(normalized['target_valid'], preds) 

                    self.update_best_model(svr, error)
    
    def get_predict(self, normalized):
        
        
        pred_test = self.best_svr.predict(normalized['input_test_norm'])
        error_test = mean_squared_error(normalized['target_test'], pred_test)
        self.lst_results.append(error_test)

        return {
            'pred_test': pred_test,
            'error_test': error_test
        }


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
            'input_test': normalized['input_test'],
            'target_test': normalized['target_test'],
            'best_rna': self.best_rna,
            'best_errors_list': self.best_errors_list
        }
    
def forecast_svr(X, y, C_values, epsilon_values, gamma_values):

    lst_results =[]

    X_train, X_valid, X_test, y_train, y_valid, y_test = prc.split_dataset(X, y)

    X_train_norm, X_valid_norm, X_test_norm = prc.norm_X_dataset(X_train, X_valid, X_test)

    # SVR é sensível à escala do X, mas não precisa normalizar y
    # y_train_norm, y_valid_norm = prc.norm_y_dataset(y_train, y_valid)

    

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


