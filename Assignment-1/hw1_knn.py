from __future__ import division, print_function

from typing import List, Callable

import numpy
import scipy


############################################################################
# DO NOT MODIFY ABOVE CODES
############################################################################

class KNN:

    def __init__(self, k: int, distance_function) -> float:
        self.k = k
        self.distance_function = distance_function

    def train(self, features: List[List[float]], labels: List[int]):
        self.training_features = numpy.asarray(features)
        self.training_label = numpy.asarray(labels)
        #raise NotImplementedError

    def predict(self, features: List[List[float]]) -> List[int]:
        label_pre = []
        for feature_pre in features:
            distance = numpy.empty(self.k,dtype=float)
            label = numpy.empty(self.k,dtype=int)
            for i in range (0,self.k):
                distance[i] = self.distance_function(feature_pre,self.training_features[i])
                label[i] = self.training_label[i]
            for i in range (self.k, len(self.training_features)):
                dis = self.distance_function(feature_pre,self.training_features[i])
                if dis < numpy.amax(distance):
                    index = numpy.argmax(distance)
                    distance[index]=dis
                    label[index]=self.training_label[i]                    
            flag = 0
            for i in range (0,self.k):
                if label[i] == 1:
                    flag += 1
                else:
                    flag -= 1
            if flag >= 0:
                label_pre.append(1)
            else:
                label_pre.append(0)
        return label_pre
        
        #raise NotImplementedError


if __name__ == '__main__':
    print(numpy.__version__)
    print(scipy.__version__)
