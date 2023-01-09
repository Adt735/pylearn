from pylearn import utils
from tabulate import tabulate
from itertools import combinations
import numpy as np
import pandas as pd

class LinearRegression():
    """
    LinearRegression fits a linear model with coefficients w = (w1, …, wp) 
    to minimize the residual sum of squares between the observed targets in the dataset, 
    and the targets predicted by the linear approximation.
    """
    def __init__(self, add_column_ones=True, keys=None, add_key_intersect=True):
        """
        Params:
        * add_column_ones(bool): Whether or not is necessary to add a column of ones (to calculate the intersect)
        * keys (list): Column Names of the dataset (name of the variables)
        * add_key_intersect(bool): Wether the first key indicates the intersection or not
        """
        #Save information about the 'X' Matrix
        self.add_column_ones = add_column_ones

        #Load keys (if passed) and add 'const' key if necessary
        self.keys = keys if keys is None else np.array([keys])
        self.add_key_intersect = add_key_intersect
        if self.keys is not None and add_key_intersect:
            self.keys = np.append([["const"]], self.keys, 1)

        #Load Params as None
        self.X = None
        self.y = None

        self.coefs = np.array([])

        self.R2 = None
        self.adj_R2 = None


    def fit(self, X, Y):
        """
        Fit linear model. (Modifies the object)

        Params:
        * X(array-like) of shape (n_samples, n_features): Training data.
        * y(array-like) of shape (n_samples,): Target values.
        """
        # k = len(X)
        # x = 1/k * sum(X)
        # x2 = 1/k * sum([i**2 for i in X])
        # y = 1/k * sum(Y)
        # xy = 1/k * sum([i*j for i,j in zip(X,Y)])

        # self.a1 = (xy - x*y) / (x2 - (x)**2)
        # self.a0 = y - self.a1*x
        # ------------------------------------------------------------

        # Load params
        self.X = X
        self.y = Y
        self.coefs = np.array([])

        # Create the 'x' matrix (adding a column of ones if needed)
        x = np.matrix(X)
        if self.add_column_ones:
            x = np.append(np.ones([X.shape[0], 1]), x, 1)
        x = np.matrix(x, dtype="float64")
        
        # Computing the 'A' matrix to have a Compatible Determinated System
        x_matrix = x.transpose() * x
        # x_matrix_inv = np.linalg.inv(x_matrix) #---------------------><<<<<<<

        # Computing the 'y' vector through the formula
        y = np.array([sum([Y[j] * x[j, i] for j in range(x.shape[0])]) for i in range(x.shape[1])], dtype="float64")

        # Solving the system
        # coefs = np.array((np.matrix(x_matrix_inv, dtype="float64") * np.matrix(y, dtype="float64").transpose()).flatten()) -> Is the same, but with less precision
        coefs = np.array([np.linalg.solve(x_matrix, y)])

        # Saving the coeficients (adding a 0 if no added 1's column) (to calculate predictions)
        self.coefs = coefs

        # Computing statistical params
        self.R2 = utils.calc_R2(Y, self.predict(X))
        self.adj_R2 = utils.calc_adj_R2(Y, self.predict(X), X.shape[1])

        if self.keys is None:
            cols = X.shape[1] if not self.add_column_ones else X.shape[1]+1
            self.keys = np.array([["const"] + [f"x{i}" for i in range(1, cols)]])


    def predict(self, X):
        """
        Predict using the linear model.

        Params:
        * X(array-like) of shape (n_samples, n_features): Samples

        Returns:
        * Y(array-like) of shape (n_samples,): Predicted Values.
        """
        X = np.array(X)

        # Modifying the coeffincients if we hadn't added a column of ones
        coefs = np.append(np.array([[0]], dtype="float64"), self.coefs, 1) if not self.add_column_ones else self.coefs

        #Coputing the predictions
        return [coefs[0, 0] + sum([coefs[0, i+1] * X[j, i] for i in range(X.shape[1])]) for j in range(X.shape[0])]  
        

    def best_adj_R2_transform(self):
        """
        Gets rid of unnecessary variables that does't have a significantly statistical weight

        Returns:
        * LR: A LinearRegression object fitted with the necessary vars
        """
        y = self.y

        # Loading a new Linear Regression object
        new_lin_reg = LinearRegression(self.add_column_ones)

        # Saving the current best params (current params)
        best_X, best_adj_R2 = self.X, self.adj_R2
        new_best_X, new_best_adj_R2 = list(range(best_X.shape[1])), self.adj_R2

        #Ensuring we enter the loop
        millorat = True

        while millorat:
            millorat = False

            #List of all possible combinations without one variable
            possible_X = list(combinations(list(range(best_X.shape[1])), best_X.shape[1]-1))

            for new_X in possible_X:
                # Fit the Model for each combination
                new_lin_reg.fit(best_X[:, new_X], y)

                # Update best model and ensure we repeat the loop without these variable
                if new_lin_reg.adj_R2 >= best_adj_R2 and new_lin_reg.adj_R2 >= new_best_adj_R2:
                    millorat = True
                    new_best_X, new_best_adj_R2 = new_X, new_lin_reg.adj_R2
            
            # Update correctly the params
            best_X, best_adj_R2 = self.X[:, new_best_X], new_best_adj_R2

        # Create the best Linear model, fit it and return it
        keys_index = [0] + [i+1 for i in new_best_X] if self.add_column_ones else new_best_X
        best_lin_reg = LinearRegression(self.add_column_ones, self.keys[0, (keys_index)], False)
        best_lin_reg.fit(best_X, y)
        return best_lin_reg


    def print_info(self):
        """
        Prints the information about the model
        """
        table_coefs = tabulate(self.coefs, headers=self.keys[0], tablefmt='fancy_grid')

        mse = utils.calc_se(self.y, self.predict(self.X))/len(self.y)
        errors = [[mse], [self.R2], [self.adj_R2]]
        table_error = utils.table_statistical_info(errors, ["Mean Squared Error", "R^2", "Adj R^2"])

        print(table_coefs)
        print(table_error)

