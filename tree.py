from pylearn import utils
import numpy as np
import pandas as pd

class DecisionTreeRegressor():
    """
    A decision tree regressor
    """
    def __init__(self, min_samples_split=32, max_depth=64, depth=0, node_type="root", rule=""):
        """
        Params:
        min_samples_split(int, default=32): The minimum number of samples required to split an internal node
        max_depth(int, default=64): The maximum depth of the tree
        """
        #Saving the hyper parameters
        self.min_samples_split = min_samples_split

        # Saving the current and max depth
        self.max_depth = max_depth
        self.depth = depth

        # Saving the node type and the rule used to get there
        self.node_type = node_type
        self.rule = rule

        # Initiating the left and right nodes as empty nodes
        self.left = None 
        self.right = None 

        # Default values for splits
        self.best_feature = None 
        self.best_value = None 


    @staticmethod
    def get_mse(ytrue, yhat):
        """
        Method to calculate the mean squared error 
        """
        # Getting the total number of samples
        n = len(ytrue)

        # Computing and returning the mean squared error
        return np.sum((ytrue - yhat)**2) / n

    @staticmethod
    def ma(x: np.array, window: int):
        """
        Calculates the moving average of the given list.
        Example: [1,2,3] -> [1.5, 2.5]
        """
        return np.array([x[i] + x[i+1] for i in range(len(x)-1)]) / 2


    def load_properties(self, X, y):
        """
        Save all the properties to the DecisionRegressorTree instance
        """
        #Loading the properties of each node
        self.X = X
        self.y = y

        self.features = list(range(X.shape[1]))

        self.n = len(y)
        self.ymean = np.mean(y)

        self.mse = self.get_mse(y, self.ymean)
        

    def best_split(self):
        """
        Given the X features and Y targets calculates the best split 
        for a decision tree.

        Returns:
        * best_feature(int): Feature to make the split
        * best_value(int): Value where the best split would take action
        """
        # Getting the GINI impurity for the base input 
        mse_base = self.mse

        # Default best feature and split
        best_feature = None
        best_value = None

        for feature in self.features:
            # Sorting the values and getting the rolling average
            X_sorted = np.sort(self.X[:, feature])
            xmeans = self.ma(np.unique(X_sorted), 2)

            for value in xmeans:
                # Getting the left and right ys 
                left_y = self.y[self.X[:, feature] < value]
                right_y = self.y[self.X[:, feature] < value]

                # Calculating the mse 
                mse_split = (self.get_mse(left_y, np.mean(left_y)) + self.get_mse(right_y, np.mean(right_y))) / 2

                # Checking if this is the best split so far 
                if mse_split < mse_base:
                    best_feature = feature
                    best_value = value 

                    # Setting the best gain to the current one 
                    mse_base = mse_split

        return (best_feature, best_value)


    def fit(self, X, y):
        """
        Recrsive method to create the decision tree
        It modifies the DecisionRegressorTree instance

        Params:
        * X(array-like) of shape (n_samples, n_features): Training data.
        * y(array-like) of shape (n_samples,): Target values.
        """

        self.load_properties(X, y)

        if (self.depth < self.max_depth) and (self.n >= self.min_samples_split):

            #Getting the best split
            best_feature, best_value = self.best_split()
            
            if best_feature is not None:
                # Saving the best split to the current node 
                self.best_feature = best_feature
                self.best_value = best_value

                # Getting the left and right nodes
                left_values = self.X[:, self.best_feature] <= self.best_value
                left_X, left_y = self.X[left_values], self.y[left_values]
            
                right_values = np.logical_not(left_values)
                right_X, right_y = self.X[right_values], self.y[right_values]

                # Creating the left and right nodes
                left = DecisionTreeRegressor( 
                    depth=self.depth + 1, 
                    max_depth=self.max_depth, 
                    min_samples_split=self.min_samples_split, 
                    node_type='left_node',
                    rule=f"{best_feature} <= {round(best_value, 3)}"
                    )

                self.left = left 
                self.left.fit(left_X, left_y)

                right = DecisionTreeRegressor( 
                depth=self.depth + 1, 
                max_depth=self.max_depth, 
                min_samples_split=self.min_samples_split, 
                node_type='right_node',
                rule=f"{best_feature} > {round(best_value, 3)}"
                )

                self.right = right 
                self.right.fit(right_X, right_y)

        # Computing statistical params
        self.R2 = utils.calc_R2(y, self.predict(X))
        self.adj_R2 = utils.calc_adj_R2(y, self.predict(X), X.shape[1])


    def predict(self, X):
        """
        Predict regression value for X.

        ## Params:
        ----------
        * X(array-like) of shape (n_samples, n_features): Samples

        ## Returns:
        -----------
        * Y(array-like) of shape (n_samples,): Predicted Values.
        """

        # Load correctly the matrix
        X = np.array(X)

        # Create a predictions array
        y_pred = np.ones((X.shape[0], 1))

        # Follow all the paths to predict correctly the values
        for i in range(X.shape[0]):
            node = self
            while not (node.left is None and node.right is None):
                if X[i, node.best_feature] <= node.best_value:
                    node = node.left
                else:
                    node = node.right
            y_pred[i] = node.ymean

        return y_pred


    def print_info(self, width=4):
        """
        Method to print the infromation about the tree
        """
        # Defining the number of spaces 
        const = int(self.depth * width ** 1.5)
        spaces = "-" * const
        
        if self.node_type == 'root':
            mse = utils.calc_se(self.y, self.predict(self.X))/len(self.y)
            errors = [[mse], [self.R2], [self.adj_R2]]
            table_error = utils.table_statistical_info(errors, ["Mean Squared Error", "R^2", "Adj R^2"])

            print(table_error)
            print("Root")
        else:
            print(f"|{spaces} Split rule: {self.rule}")
        print(f"{' ' * const}   | MSE of the node: {round(self.mse, 2)}")
        print(f"{' ' * const}   | Count of observations in node: {self.n}")
        print(f"{' ' * const}   | Prediction of node: {round(self.ymean, 3)}")   

    def print_tree(self):
        """
        Prints the whole tree from the current node to the bottom
        """
        self.print_info() 
        
        if self.left is not None: 
            self.left.print_tree()
        
        if self.right is not None:
            self.right.print_tree()
