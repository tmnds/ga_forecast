from sklearn.metrics import mean_squared_error
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor

from conf.base_model import BaseModel
class RF(BaseModel):
    '''
        Comentário sobre o Random Forest
    '''
    def __init__(self,
                n_estimators_values = [50, 100, 200],
                max_depth_values = [None, 5, 10, 20]
                ):

        super().__init__()

        self.n_estimators_values = n_estimators_values
        self.max_depth_values = max_depth_values

    def grid_search(self, data):

        for n in self.n_estimators_values:
            for d in self.max_depth_values:

                rf = RandomForestRegressor(n_estimators=n, max_depth=d, random_state=self.current_seed)
                rf.fit(data['input_train_norm'], data['target_train'])

                preds = rf.predict(data['input_valid_norm'])
                error = mean_squared_error(data['target_valid'], preds)

                self.update_best_model(rf, error)
    
    def get_predict(self, data):
        
        pred_test = self.best_model.predict(data['input_test_norm'])
        error_test = mean_squared_error(data['target_test'], pred_test)
        self.lst_results.append(error_test)

        return {
            'pred_test': pred_test,
            'error_test': error_test
        }

    def train(self, data):

        seeds = 10

        for sd in range(seeds):

            self.current_seed = sd
            self.grid_search(data)   
            predict = self.get_predict(data) 
        
        return {
            
            'lst_results': self.lst_results,
            'pred_test': predict['pred_test'],
            'best_rf': self.best_model,
            'best_errors_list': self.best_errors_list
        }
class GBoosting(BaseModel):
    '''
        Comentário Sobre o Gradient Boosting

    '''
    def __init__(self, 
                n_estimators_values = [50, 100, 200],
                max_depth_values = [2, 3, 5],
                learning_rate_values = [0.01, 0.05, 0.1]
                ):

        super().__init__()

        self.n_estimators_values = n_estimators_values
        self.max_depth_values = max_depth_values
        self.learning_rate_values = learning_rate_values

    def grid_search(self, data):

        for n in self.n_estimators_values:
            for d in self.max_depth_values:
                for lr in self.learning_rate_values:

                    gb = GradientBoostingRegressor(n_estimators=n, max_depth=d, learning_rate=lr, random_state=self.current_seed)
                    gb.fit(data['input_train_norm'], data['target_train'])

                    preds = gb.predict(data['input_valid_norm'])
                    error = mean_squared_error(data['target_valid'], preds)

                    self.update_best_model(gb, error)
    
    def get_predict(self, data):
        
        pred_test = self.best_model.predict(data['input_test_norm'])
        error_test = mean_squared_error(data['target_test'], pred_test)
        self.lst_results.append(error_test)

        return {
            'pred_test': pred_test,
            'error_test': error_test
        }

    def train(self, data):

        seeds = 10

        for sd in range(seeds):

            self.current_seed = sd
            self.grid_search(data)   
            predict = self.get_predict(data) 
        
        return {
            'lst_results': self.lst_results,
            'pred_test': predict['pred_test'],
            'best_gb': self.best_model,
            'best_errors_list': self.best_errors_list
        }