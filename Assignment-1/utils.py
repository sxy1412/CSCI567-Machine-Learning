from typing import List

import numpy as np
import copy


def mean_squared_error(y_true: List[float], y_pred: List[float]) -> float:
    assert len(y_true) == len(y_pred)
    mse = 0
    for y_t,y_p in zip(y_true,y_pred):
        mse += np.square(y_t-y_p)
    return mse/len(y_true)


def f1_score(real_labels: List[int], predicted_labels: List[int]) -> float:
    """
    f1 score: https://en.wikipedia.org/wiki/F1_score
    """
    assert len(real_labels) == len(predicted_labels)
    correct_positive = 0
    all_positive_pre = 0
    all_positive_real = 0
    for r,p in zip(real_labels, predicted_labels):
        if r == 1:
            all_positive_real += 1
            if p == 1:
                correct_positive += 1
                all_positive_pre += 1
        else:
            if p == 1:
                all_positive_pre += 1
    if all_positive_pre == 0 or all_positive_real == 0:
        return 0
    precision = correct_positive/all_positive_pre
    recall = correct_positive/all_positive_real
    if precision+recall == 0:
        return 0
    return 2*precision*recall/(precision+recall)


def polynomial_features(
        features: List[List[float]], k: int
) -> List[List[float]]:
    n = len(features)
    feats = copy.deepcopy(features)
    for feat in feats:
        m = len(feat)
        for i in range(2,k+1):
            for j in range(0,m):
                feat.append(np.power(feat[j],i))
    return feats


def euclidean_distance(point1: List[float], point2: List[float]) -> float:
    sub_list = []
    for p1,p2 in zip(point1,point2):
        sub_list.append(p1-p2)
    return np.sqrt(inner_product_distance(sub_list,sub_list))


def inner_product_distance(point1: List[float], point2: List[float]) -> float:
    inner_dis = 0
    for p1,p2 in zip(point1,point2):
        inner_dis += p1*p2
    return inner_dis


def gaussian_kernel_distance(
        point1: List[float], point2: List[float]
) -> float:
    sum_of_dif_sqrt = 0
    for p1,p2 in zip(point1,point2):
        sum_of_dif_sqrt += np.power(p1-p2,2)
    gkd = -np.exp(-0.5*sum_of_dif_sqrt)
    return gkd


class NormalizationScaler:
    def __init__(self):
        pass

    def __call__(self, features: List[List[float]]) -> List[List[float]]:
        """
        normalize the feature vector for each sample . For example,
        if the input features = [[3, 4], [1, -1], [0, 0]],
        the output should be [[0.6, 0.8], [0.707107, -0.707107], [0, 0]]
        """
        normalized = []
        for feature in features:
            denominator = np.sqrt(inner_product_distance(feature,feature))
            if denominator == 0:
                normalized.append([0]*len(feature))
            else:
                normalized_vector = []
                for x in feature:
                    num = np.around(x/denominator,decimals=6)
                    normalized_vector.append(num)
                normalized.append(normalized_vector)
        return normalized


class MinMaxScaler:
    """
    You should keep some states inside the object.
    You can assume that the parameter of the first __call__
        must be the training set.

    Note:
        1. you may assume the parameters are valid when __call__
            is being called the first time (you can find min and max).

    Example:
        train_features = [[0, 10], [2, 0]]
        test_features = [[20, 1]]

        scaler = MinMaxScale()
        train_features_scaled = scaler(train_features)
        # now train_features_scaled should be [[0, 1], [1, 0]]

        test_features_sacled = scaler(test_features)
        # now test_features_scaled should be [[10, 0.1]]

        new_scaler = MinMaxScale() # creating a new scaler
        _ = new_scaler([[1, 1], [0, 0]]) # new trainfeatures
        test_features_scaled = new_scaler(test_features)
        # now test_features_scaled should be [[20, 1]]

    """

    def __init__(self):
        self.flag = 0
        self.min_in_col = []
        self.max_in_col = []
        pass

    def __call__(self, features: List[List[float]]) -> List[List[float]]:
        """
        normalize the feature vector for each sample . For example,
        if the input features = [[2, -1], [-1, 5], [0, 0]],
        the output should be [[1, 0], [0, 1], [0.333333, 0.16667]]
        """
        feature_matrix = np.asarray(features)
        n,k = feature_matrix.shape
        scaled_matrix = np.arange(n*k,dtype=float).reshape(n,k)
        for j in range (0,k):
            if self.flag ==0:
                maximum = feature_matrix[:,j].max()
                minimum = feature_matrix[:,j].min()
                self.min_in_col.append(minimum)
                self.max_in_col.append(maximum)
            for i in range (0,n):
                scaled_matrix[i][j] = (feature_matrix[i][j] - self.min_in_col[j])/(self.max_in_col[j]-self.min_in_col[j])
        if self.flag == 0:
            self.flag = 1
        return scaled_matrix.tolist()   
