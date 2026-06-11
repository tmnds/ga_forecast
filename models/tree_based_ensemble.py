import processing as prc

from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor


class RF:
    def __init__(self, 
                input_windows, 
                target_values,
                n_estimators_values = [50, 100, 200],
                max_depth_values = [None, 5, 10, 20]
                ):

        self.input_windows = input_windows
        self.target_values = target_values
        
        self.n_estimators_values = n_estimators_values
        self.max_depth_values = max_depth_values

        self.lst_results = []
        self.best_errors_list = []
        self.best_error = float('inf')
        self.best_rf = None

    def normalize_data(self):
        input_train, input_valid, input_test, target_train, target_valid, target_test = prc.split_dataset(self.input_windows, self.target_values)

        input_train_norm, input_valid_norm, input_test_norm = prc.norm_X_dataset(input_train, input_valid, input_test)

        return {
            'input_test': input_test,
            'input_train_norm': input_train_norm,
            'input_valid_norm': input_valid_norm,
            'input_test_norm': input_test_norm,
            'target_train': target_train,
            'target_valid': target_valid,
            'target_test': target_test
        }
    
    def update_best_model(self, rf, error):
        
        if error < self.best_error:

            self.best_rf = rf
            self.best_error = error
            
            self.best_errors_list.append({'erro': error, 'params': self.best_rf.get_params()})

    def grid_search(self, normalized):

        for n in n_estimators_values:
            for d in max_depth_values:

                rf = RandomForestRegressor(n_estimators=n, max_depth=d, random_state=42)
                rf.fit(normalized['input_train_norm'], normalized['target_train'])

                preds = rf.predict(normalized['input_valid_norm'])
                error = mean_squared_error(normalized['target_valid'], preds)

                self.update_best_model(rf, error)
    
    def get_predict(self, normalized):
        
        pred_test = self.best_rf.predict(normalized['input_test_norm'])
        error_test = mean_squared_error(normalized['target_test'], pred_test)
        self.lst_results.append(error_test)

        return {
            'pred_test': pred_test,
            'error_test': error_test
        }

    def train_model(self):
        
        normalized = self.normalize_data()

        self.grid_search(normalized)   
        predict = self.get_predict(normalized) 
        
        return {
            
            'lst_results': self.lst_results,
            'pred_test': predict['pred_test'],
            'input_test': normalized['input_test'],
            'target_test': normalized['target_test'],
            'best_rf': self.best_rf,
            'best_errors_list': self.best_errors_list
        }

# DESCONTINUADO
# def forecast_rf(X, y, n_estimators_values, max_depth_values):

#     lst_results =[]

#     X_train, X_valid, X_test, y_train, y_valid, y_test = prc.split_dataset(X, y)

#     X_train_norm, X_valid_norm, X_test_norm = prc.norm_X_dataset(X_train, X_valid, X_test)

#     # RF não é sensível à escala — normalização opcional, mas mantém consistência
#     # y_train_norm, y_valid_norm = prc.norm_y_dataset(y_train, y_valid)

#     best_error = float('inf')
#     best_rf = None

#     for n in n_estimators_values:
#         for d in max_depth_values:

#             rf = RandomForestRegressor(n_estimators=n, max_depth=d, random_state=42)
#             rf.fit(X_train_norm, y_train)

#             preds = rf.predict(X_valid_norm)
#             error = mean_squared_error(y_valid, preds)

#             if error < best_error:
#                 best_rf = rf
#                 best_error = error

#     pred_test = best_rf.predict(X_test_norm)
#     error_test = mean_squared_error(y_test, pred_test)
#     lst_results.append(error_test)
    
#     return lst_results, pred_test, X_test, y_test, best_rf
