from __future__ import division, print_function

from typing import List

import numpy
import scipy


############################################################################
# DO NOT MODIFY ABOVE CODES
############################################################################

class LinearRegression:
    def __init__(self, nb_features: int):
        self.nb_features = nb_features

    def train(self, features: List[List[float]], values: List[float]):
        feature_matrix = numpy.asarray(features)
        n,m = feature_matrix.shape
        X0 = numpy.ones((n,1))
        X = numpy.hstack((X0,feature_matrix))
        Y = numpy.asarray(values)
        self.weight=numpy.dot(numpy.dot(numpy.linalg.inv((numpy.dot(X.T,X))),X.T),Y)
        #print(self.weight.shape)
        #raise NotImplementedError

    def predict(self, features: List[List[float]]) -> List[float]:
        w0,w=numpy.split(self.weight,[1,])
        feature_matrix = numpy.asarray(features)
        y=numpy.dot(feature_matrix,w)+w0
        #print(y)
        return y.tolist()
        #raise NotImplementedError

    def get_weights(self) -> List[float]:
        return self.weight.tolist()

        """
        for a model y = 1 + 3 * x_0 - 2 * x_1,
        the return value should be [1, 3, -2].
        """
        #raise NotImplementedError


class LinearRegressionWithL2Loss:
    '''Use L2 loss for weight regularization'''
    def __init__(self, nb_features: int, alpha: float):
        self.alpha = alpha
        self.nb_features = nb_features

    def train(self, features: List[List[float]], values: List[float]):
        feature_matrix = numpy.asarray(features)
        n,m = feature_matrix.shape
        X0 = numpy.ones((n,1))
        X = numpy.hstack((X0,feature_matrix))
        Y = numpy.asarray(values)
        alpha_matrix = numpy.zeros((m+1,m+1),float)
        numpy.fill_diagonal(alpha_matrix, self.alpha)
        self.weight=numpy.dot(numpy.dot(numpy.linalg.inv((numpy.dot(X.T,X))+alpha_matrix),X.T),Y)
        #raise NotImplementedError

    def predict(self, features: List[List[float]]) -> List[float]:
        w0,w=numpy.split(self.weight,[1,])
        feature_matrix = numpy.asarray(features)
        y=numpy.dot(feature_matrix,w)+w0
        #print(y)
        return y.tolist()
        #raise NotImplementedError

    def get_weights(self) -> List[float]:
        return self.weight.tolist()
        """
        for a model y = 1 + 3 * x_0 - 2 * x_1,
        the return value should be [1, 3, -2].
        """
        #raise NotImplementedError


if __name__ == '__main__':
    print(numpy.__version__)
    print(scipy.__version__)
