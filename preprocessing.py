import numpy as np
import pandas as pd

class LabelEncoder():
    def __init__(self):
        pass

    def fit_transform(self, column):
        elements = list(set(column))
        new_column_dict = {element:i for i, element in enumerate(elements)}
        return pd.Series([new_column_dict[i] for i in column])


class OneHotEncoder():
    def __init__(self, categorical_features=[3], drop=True):
        self.categorical_features = categorical_features
        self.drop = drop
        pass

    def fit_transform(self, dataset):
        for category in self.categorical_features:
            defferent_nums = list(set(dataset[:, category]))
            for num in defferent_nums:
                # print(([1  if i==num else 0 for i in dataset[:, category]]))
                dataset = np.append(dataset, np.array([[1]  if i==num else [0] for i in dataset[:, category]]), axis=1)
            dataset = np.delete(dataset, category, 1)
            if self.drop:
                dataset = np.delete(dataset, -1, 1)
        return dataset


class PolynomialFeatures():
    def __init__(self, degree=2):
        self.degree = degree
    
    def fit_transform(self, X):
        matrix = [[X.reshape(1, len(X)).tolist()[0][j]**i for i in range(self.degree+1)] for j in range(len(X))]
        return np.array(matrix)


class StandardScaler():
    def __init__(self):
        pass

    # calculate column means
    @staticmethod
    def column_means(dataset):
        means = [0 for _ in range(len(dataset[0]))]
        for i in range(len(dataset[0])):
            means[i] = sum(dataset[:, i]) / len(dataset)
        return np.array(means)
    
    # calculate column standard deviations
    @staticmethod
    def column_stdevs(dataset, means):
        stdevs = [0 for _ in range(len(dataset[0]))]
        for i in range(len(dataset[0])):
            stdevs[i] = sum((dataset[:, i] - means[i])**2)
        return np.sqrt(np.array(stdevs) / (len(dataset)-1))

    def fit_transform(self, dataset):
        self.fit(dataset)
        return self.transform(dataset)

    def fit(self, dataset):
        self.means = self.column_means(dataset)
        self.stdevs = self.column_stdevs(dataset, self.means)

    def transform(self, dataset):
        return (dataset - self.means) / self.stdevs

    def inverse_transform(self, dataset):
        return dataset * self.stdevs + self.means
