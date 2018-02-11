from __future__ import division, print_function

from typing import List, Tuple, Callable

import numpy as np
import scipy
import matplotlib.pyplot as plt
import copy

class Perceptron:

    def __init__(self, nb_features=2, max_iteration=10, margin=1e-4):
        '''
            Args : 
            nb_features : Number of features
            max_iteration : maximum iterations. You algorithm should terminate after this
            many iterations even if it is not converged 
            margin is the min value, we use this instead of comparing with 0 in the algorithm
        '''
        
        self.nb_features = 2
        self.w = [0 for i in range(0,nb_features+1)]
        self.margin = margin
        self.max_iteration = max_iteration

    def train(self, features: List[List[float]], labels: List[int]) -> bool:
        '''
            Args  : 
            features : List of features. First element of each feature vector is 1 
            to account for bias
            labels : label of each feature [-1,1]
            
            Returns : 
                True/ False : return True if the algorithm converges else False. 
        '''
        ############################################################################
        # TODO : complete this function. 
        # This should take a list of features and labels [-1,1] and should update 
        # to correct weights w. Note that w[0] is the bias term. and first term is 
        # expected to be 1 --- accounting for the bias
        ############################################################################
        weight = np.asarray(self.w,dtype = float) # transposed weight vector
        weight[0] = 1
        i=0
        x = copy.deepcopy(features)
        y = copy.deepcopy(labels)
        for i in range(0,self.max_iteration):
            z = list(zip(x,y))
            np.random.shuffle(z)
            x,y = zip(*z)
            for feat,label in zip(x,y): #update rule
                feature = np.asarray(feat)
                y_real = label
                y_pre = np.sign(np.dot(weight,feature.T))
                if y_real != y_pre:
                    weight += y_real*feature/np.linalg.norm(feature)
            converge = 1
            for feat,label in zip(x,y): #check convergence
                feature = np.asarray(feat)
                y_real = label
                y_pre = np.sign(np.dot(weight,feature.T))
                if y_real != y_pre:
                    converge = 0
                    break
            if converge == 1:
                break
        self.w = weight.tolist()
        if converge == 1:
            return True
        else:
            return False
            
#         weight = np.asarray(self.w,dtype = float) # transposed weight vector
#         weight[0] = 1
#         i = 0
#         while True:
#             i += 1
#             sample_number = np.random.randint(0,len(features))
#             selected_sample = np.asarray(features[sample_number])
#             y_real = labels[sample_number]
#             y_pre = np.sign(np.dot(weight,selected_sample.T))
#             if y_real != y_pre:
#                 weight += y_real*selected_sample/np.linalg.norm(selected_sample)

#             if i >= self.max_iteration:
#                 break
#         self.w = weight.tolist() 
#         for sample,label in zip(features,labels):# if converged?
#             y_pre = np.sign(np.dot(weight,np.asarray(sample).T))
#             if label != y_pre:
#                 return False
#         return True
        #raise NotImplementedError
    
    def reset(self):
        self.w = [0 for i in range(0,self.nb_features+1)]
        
    def predict(self, features: List[List[float]]) -> List[int]:
        '''
            Args  : 
            features : List of features. First element of each feature vector is 1 
            to account for bias
            
            Returns : 
                labels : List of integers of [-1,1] 
        '''
        ############################################################################
        # TODO : complete this function. 
        # This should take a list of features and labels [-1,1] and use the learned 
        # weights to predict the label
        ############################################################################
        y = []
        weight = np.asarray(self.w)
        for sample in features:
            y_pre = np.sign(np.dot(weight,np.asarray(sample).T))
            y.append(y_pre)
        return y
        #raise NotImplementedError

    def get_weights(self) -> Tuple[List[float], float]:
        return self.w
    
