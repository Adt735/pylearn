from pylearn.tree import DecisionTreeRegressor
from pylearn import utils
import numpy as np


class RandomForestRegressor():
    """
    A random forest regressor with n_estimators `DecisionTreeRegressor`s
    """
    def __init__(self, n_estimators=100, min_observations=1/2, min_samples_split=20, max_depth=64):
        """
        ## Params:
        ----------
        * n_estimators(int, default=100): The number of trees in the forest.
        * min_observations(float, [0, 1]): Minimum number of observations to be part of each tree
        * min_samples_split(int, default=20): The minimum number of samples required to split an internal node
        * max_depth(int, default=64): The maximum depth of each tree
        """
        self.n_estimators = n_estimators
        self.min_observations = min_observations
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth

        # Creating vars for observations
        self.X = None
        self.y = None

        self.trees = [DecisionTreeRegressor(self.min_samples_split, self.max_depth) for _ in range(self.n_estimators)]

    @staticmethod
    def get_random_arrays_of_observations(a, b, min_obs):
        """
        Returns two arrays shuffled in unison and with a random number of observations,
        ranging from `total_obs * min_observations` to `total_obs`
        """
        assert len(a) == len(b)
        p_ = np.random.permutation(len(a))
        nums = sorted([np.random.randint(0, len(a)-1), np.random.randint(0, len(a)-1)])

        while abs(nums[0] - nums[1]) < len(a) * min_obs:
            nums = sorted([np.random.randint(0, len(a)-1), np.random.randint(0, len(a)-1)])
        p = p_[nums[0]:nums[1]]

        return a[p, :], b[p]


    def fit(self, X, y):
        """
        Build a forest of trees from the training set (X, y)
        It modifies the RandomForestRegressor instance

        ## Params:
        ------------
        * X(array-like) of shape (n_samples, n_features): Training data.
        * y(array-like) of shape (n_samples,): Target values.
        """
        # Saving the observations arrays
        self.X = X
        self.y = y

        for tree in self.trees:
            shuffled_X, shuffled_y = self.get_random_arrays_of_observations(X, y, self.min_observations)
            tree.fit(shuffled_X, shuffled_y)

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
        ------------
        * Y(array-like) of shape (n_samples,): Predicted Values.
        """
        X = np.array(X)
        return sum([tree.predict(X) for tree in self.trees]) / len(self.trees)


    def print_info(self):
        """
        Prints the information about the model
        """
        mse = utils.calc_se(self.y, self.predict(self.X))/len(self.y)
        errors = [[mse], [self.R2], [self.adj_R2]]
        table_error = utils.table_statistical_info(errors, ["Mean Squared Error", "R^2", "Adj R^2"])

        print(table_error)
