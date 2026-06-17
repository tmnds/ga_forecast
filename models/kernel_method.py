import conf.processing as prc

from sklearn.metrics import mean_squared_error
from sklearn.svm import SVR

class SVM:
    '''
        Comentário sobre o Support Vector Machine
    
    '''
    def __init__(self, 
                C_values=[0.1, 1, 10, 100], # Penalidade por erro
                epsilon_values = [0.01, 0.05, 0.1],
                gamma_values = ['scale', 0.01, 0.1]
                ):

        self.C_values = C_values
        self.epsilon_values = epsilon_values
        self.gamma_values = gamma_values

        self.lst_results = []
        self.best_errors_list = []
        self.best_error = float('inf')
        self.best_svr = None

    def update_best_model(self, svr, error):
        
        if error < self.best_error:

            self.best_svr = svr
            self.best_error = error
            self.best_errors_list.append({'erro': error, 'params': self.best_svr.get_params()})

    def get_predict(self, data):
        
        pred_test = self.best_svr.predict(data['input_test_norm'])
        error_test = mean_squared_error(data['target_test'], pred_test)
        self.lst_results.append(error_test)

        return {
            'pred_test': pred_test,
            'error_test': error_test
        }

    def grid_search(self, data):

        for c in self.C_values:
            for e in self.epsilon_values:
                for g in self.gamma_values:

                    svr = SVR(kernel='rbf', C=c, epsilon=e, gamma=g)
                    svr.fit(data['input_train_norm'], data['target_train'])
                
                    preds = svr.predict(data['input_valid_norm'])
                    error = mean_squared_error(data['target_valid'], preds) 

                    self.update_best_model(svr, error)
    
    def train(self, data):
        

        self.grid_search(data)   
        predict = self.get_predict(data) 

        return {
            
            'lst_results': self.lst_results,
            'pred_test': predict['pred_test'],
            'input_test': data['input_test'],
            'target_test': data['target_test'],
            'best_svr': self.best_svr,
            'best_errors_list': self.best_errors_list
        }
