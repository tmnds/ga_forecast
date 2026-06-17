class BaseModel:
    def __init__(self):
        self.lst_results = []
        self.best_errors_list = []
        self.best_error = float('inf')
        self.best_model = None
        self.current_seed = None
    
    def update_best_model(self, model, error):
        
        if error < self.best_error:

            self.best_model = model
            self.best_error = error
            self.best_errors_list.append({'erro': error, 'params': self.best_model.get_params()})
