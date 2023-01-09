from tabulate import tabulate
import numpy as np

def calc_se(ytrue, yhat):
    """
    Method to calculate the squared error 
    """
    # Computing and returning the squared error
    return np.sum((ytrue.reshape(-1, 1) - yhat.reshape(-1, 1))**2)

def calc_R2(y_true, y_hat):
    return 1 - (calc_se(y_true, y_hat)) / calc_se(y_true, np.mean(y_true))

def calc_adj_R2(y_true, y_hat, n_vars):
    return 1 - (1 - calc_R2(y_true, y_hat)) * (len(y_true)-1) / (len(y_true) - n_vars - 1)

def table_statistical_info(errors_info, errors_index):
    return tabulate(errors_info, tablefmt='fancy_grid', showindex=errors_index)