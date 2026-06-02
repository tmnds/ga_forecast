import warnings

import numpy as np
import pandas as pd
import copy as cp

import processing as prc

from sklearn.metrics import mean_squared_error
from sklearn.neural_network import MLPRegressor
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.preprocessing import MinMaxScaler

# This will suppress ALL FutureWarning messages
warnings.simplefilter(action='ignore', category=RuntimeWarning)

def mean_square_error(y_true, y_pred):
    y_true = np.asmatrix(y_true).reshape(-1)
    y_pred = np.asmatrix(y_pred).reshape(-1)

    return np.square(np.subtract(y_true, y_pred)).mean()

def root_mean_square_error(y_true, y_pred):

    return mean_square_error(y_true, y_pred)**0.5

def sigmoid(x):
    # Função de ativação Sigmoid.
    # Transforma o valor de entrada para um valor entre 0 e 1.
    return 1 / (1 + np.exp(-x))

def linear(x):
    """
    Função de ativação Linear.
    A saída é exatamente igual à entrada (nenhuma transformação).
    """
    return x

def calculate_h(X, omega, b, act_fuction):
    """
    Calcula a saída de um único neurônio oculto (h_L(X)).
    Assumimos que omega é um vetor coluna e X é uma matriz de entrada.
    """
    # (X @ omega) realiza o produto escalar de cada linha de X com omega
    # O resultado é um vetor de saídas para cada amostra
    return act_fuction(X @ omega + b)

def transform_verify_numpy(array):
    if (
        (isinstance(array, pd.DataFrame)) or 
        (isinstance(array, pd.Series))
        ):
        array = array.to_numpy()

    if not isinstance(array, np.ndarray):
        raise NotImplementedError("inputs e outputs need to be pd.Dataframe or np.ndarray")
    
    return array

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
        input_train, input_valid, input_test, target_train, target_valid, target_test = prc.split_dataset(self.input_windows, self.target_values)

        # Normalização das janelas
        input_train_norm, input_valid_norm, input_test_norm = prc.norm_X_dataset(input_train, input_valid, input_test)
       
        # Normalização de Alvos
        target_train_norm, target_valid_norm = prc.norm_y_dataset(target_train, target_valid)
        
        return {
            'input_train_norm': input_train_norm,
            'input_valid_norm': input_valid_norm,
            'input_test_norm': input_test_norm,
            'target_train_norm': target_train_norm,
            'target_valid_norm': target_valid_norm,
            
            'input_test': input_test,

            'target_train': target_train,
            'target_test': target_test
        }
        
    def update_best_model(self, rna, error):

        if error < self.best_error:

            self.best_rna = rna
            self.best_error = error
            
            self.best_errors_list.append({'erro': error, 'params': self.best_rna.get_params()})
    
    def get_predict(self, normalized):

        pred_test = self.best_rna.predict(normalized['input_test_norm'])
        pred_test_denom = prc.denorm_data(normalized['target_train'], pred_test)
        error_test = mean_squared_error(normalized['target_test'], pred_test_denom)
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
                    
                    rna.fit(normalized['input_train_norm'],normalized['target_train_norm']) # Treina o modelo com os dados de treinamento

                    preds = rna.predict(normalized['input_valid_norm'])
                    error = mean_squared_error(normalized['target_valid_norm'], preds) 

                    self.update_best_model(rna, error)
        
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

class SCN_III(BaseEstimator, RegressorMixin):

    def __init__(self, 
                t_max: int = 1,
                l_max: int = 10,
                r: int = 0.5,
                lambda_min: float = 0.2,
                lambda_max: float = 1.0,
                lambda_steps: int = 3,
                error_min = 0,
                hidden_act_fuction = sigmoid
            ):
       
        self.t_max = t_max
        self.l_max = l_max
        self.r = r 
        self.error_min = error_min
        self.scaler_x = None
        self.scaler_y = None
        self.lambda_max = lambda_max
        self.lambda_min = lambda_min
        self.lambda_steps = lambda_steps
        self.hidden_act_fuction = hidden_act_fuction
        

    def fit(self, X, y):
        self.weights_list = []

        self.lambda_set = np.linspace(self.lambda_min,  self.lambda_max, self.lambda_steps)

        x_train = X.copy()
        y_train = y.copy()
        x_train = transform_verify_numpy(x_train)
        y_train = transform_verify_numpy(y_train)
    
        if y_train.ndim!=1:
            raise NotImplementedError("multiple target is not implemented")
        
        self.scaler_x = MinMaxScaler(feature_range=(0, 1))
        self.scaler_x = self.scaler_x.fit(x_train)
        
        # TODO: need to be refactored if evolved to multi target
        self.scaler_y = MinMaxScaler(feature_range=(0, 1))
        self.scaler_y.fit(y_train.reshape(-1, 1))
 
        x_train_norm =  self.scaler_x.transform(x_train)
        y_train_norm =  self.scaler_y.transform(y_train.reshape(-1, 1)).flatten()

        self.error_list = []
        input_size = x_train_norm.shape[1]
        e = np.array(y_train_norm.copy())
        
        efro = 1
        r = self.r
        
        n_samples = len(x_train_norm)
        t_max = self.t_max 
        l_max = self.l_max 
        l_i = 1

        self.wstar_list = []
        self.bstar_list = []
        h_l_matrix = np.empty((n_samples, 0))

        while(l_i<=l_max and efro > self.error_min):
            gamma = []
            w = []
            bias = []
            for y in self.lambda_set:
                for k in range(0, t_max):# step 4
                    wl = np.random.uniform(low=-y, high=y, size=input_size)
                    bL = np.random.uniform(low=-y, high=y, size=1)
                    hL = calculate_h(x_train_norm, wl, bL, self.hidden_act_fuction)
                    
                    uL =  (1 - r)/(l_i + 1)
                    
                    eq28 = (np.dot(e.T, hL.T)**2 / np.dot(hL.T, hL)) -  (1 - r - uL) * e.T * e
                    
                    if np.min(eq28) > 0:
                        w.append(wl)
                        bias.append(bL)
                        gamma.append(np.sum(eq28))
                
            if len(w) > 0:
                wstar = w[np.argmax(gamma)]
                bstar = bias[np.argmax(gamma)]
                
                self.wstar_list.append(wstar)
                self.bstar_list.append(bstar)
        
                hlstar = calculate_h(x_train_norm, wstar, bstar, self.hidden_act_fuction)
                h_l_matrix = np.hstack((h_l_matrix, hlstar.reshape(-1, 1)))
            else:
                r = r + np.random.uniform(low=0, high=1 - r , size=1)[0]
                continue
        
            # Garante que T tem 2 dimensões para a multiplicação de matrizes
            T_reshaped = np.array(y_train_norm).reshape(n_samples, -1) if  np.array(y_train_norm).ndim == 1 else y_train_norm
            self.beta_star = np.dot(np.linalg.pinv(h_l_matrix), T_reshaped) 
            #self.beta_star = beta_current#.flatten().tolist() 

            el = np.dot(h_l_matrix,  self.beta_star) - T_reshaped
            e = el.copy()
            l_i = l_i + 1
            efro = np.square(el).mean()
            self.error_list.append(efro)
        
    def predict(self, X):
        x_test = X.copy()
        x_test = transform_verify_numpy(x_test)
        x_test_norm =  self.scaler_x.transform(x_test)

        n_test_samples, _ = x_test_norm.shape
        #H_test = np.empty((N_test_samples, len(wstar_list))
        h_predict = np.empty((n_test_samples, 0))
        for i, (omega, b) in enumerate(zip(self.wstar_list, self.bstar_list)):
            h_result = calculate_h(x_test_norm, omega, b,  self.hidden_act_fuction)
            h_predict = np.hstack((h_predict, h_result.reshape(-1, 1)))

        predictions = np.dot(h_predict,  self.beta_star)
        predictions =  self.scaler_y.inverse_transform(predictions.reshape(-1, 1)).flatten()

        return predictions

class Model(BaseEstimator, RegressorMixin):
    model = None


    def __init__(self,
                 hidden_dim=10,
                 output_dim=1,
                 solver='sigmoid'):
        '''
        Neural Network object
        '''
        self.input_dim = None
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.solver = solver
        self.U = 0
        self.V = 0
        self.S = 0
        self.H = 0
        self.alpha = 0  # for regularization
        self.W1 = None
        self.W2 = None
    # Helper function


    def sigmoid(self, x):
        return 1.0 / (1.0 + np.exp(-0.1 * x)) - 0.5


    def predict(self, x):
        '''
        Forward pass to calculate the ouput
        '''
        x = np.matrix(x)
        if self.solver == 'sigmoid':
            hidden_activation = self.sigmoid(x @ self.W1)
        elif self.solver == 'linear':
            hidden_activation = x @ self.W1
        else:
            raise RuntimeError("solver not implemented")


        y = hidden_activation @ self.W2


        return np.array(y.flatten())[0]


    def fit(self, x, y):
        '''
        Compute W2 that lead to minimal LS
        '''

        self.input_dim = x.shape[1]
        self.W1 = np.matrix(np.random.rand(self.input_dim, self.hidden_dim))
        self.W2 = np.matrix(np.random.rand(self.hidden_dim, self.output_dim))


        X = np.matrix(x)
        Y = np.matrix(np.array(y).reshape(-1, 1))


        if self.solver == 'sigmoid':
            self.H = np.matrix(self.sigmoid(X @ self.W1))
        elif self.solver == 'linear':
            self.H = np.matrix(x @ self.W1)
        else:
            raise RuntimeError("solver not implemented")


        H = cp.deepcopy(self.H)


        self.svd(H)
        iH = np.matrix(self.V) @ np.matrix(np.diag(self.S)).I @ np.matrix(self.U).T


        self.W2 = iH * Y

        del self.H
        del self.U
        del self.S
        del self.V
        self.H = self.U = self.S = self.V = None # Garante que fiquem vazios
        
        return self


    def svd(self, h):
        '''
        Compute the Singular Value Decomposition of a matrix H
        '''
        H = np.matrix(h)
        self.U, self.S, Vt = np.linalg.svd(H, full_matrices=False)
        self.V = np.matrix(Vt).T
        return np.matrix(self.U), np.matrix(self.S), np.matrix(self.V)


# Descontinuado
# def forecast_mlp(X, y, solver, hidden_neurons, learning_rate, activation, jumps):

#     lst_results =[]

#     # Divisão do dataset e Normalizaçao
#     X_train, X_valid, X_test, y_train, y_valid, y_test = prc.split_dataset(X, y)

#     X_train_norm, X_valid_norm, X_test_norm = prc.norm_X_dataset(X_train, X_valid, X_test)
#     y_train_norm, y_valid_norm = prc.norm_y_dataset(y_train, y_valid)

#     best_errors_list = []

#     best_error = float('inf')
#     best_rna = None

#     for i in range(jumps):
#         sd=i

#         for h in hidden_neurons:
#             for l in learning_rate:
#                 for a in activation:
#                     #[NEURON ESCONDIDOS, TAXA DE APRENDIZADO, FUNCAO DE ATIVACAO]
#                     rna = MLPRegressor(hidden_layer_sizes=(h,),learning_rate_init=l,activation=a,shuffle=False, random_state=sd, solver=solver)
                    
#                     rna.fit(X_train_norm,y_train_norm) # Treina o modelo com os dados de treinamento

#                     preds = rna.predict(X_valid_norm)
#                     error = mean_squared_error(y_valid_norm, preds) 

#                     if error < best_error:
#                         best_rna = rna
#                         best_error = error
#                         best_errors_list.append({'erro': error,'params': best_rna.get_params()})

#         pred_test = best_rna.predict(X_test_norm)
#         pred_test_denom = prc.denorm_data(y_train, pred_test)
#         error_test = mean_squared_error(y_test, pred_test_denom)
#         lst_results.append(error_test)
    
#     return lst_results, pred_test, pred_test_denom, X_test, y_test, best_rna, best_errors_list

