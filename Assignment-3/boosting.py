import numpy as np
from typing import List, Set

from classifier import Classifier
from decision_stump import DecisionStump
from abc import abstractmethod

class Boosting(Classifier):
  # Boosting from pre-defined classifiers
	def __init__(self, clfs: Set[Classifier], T=0):
		self.clfs = clfs
		self.num_clf = len(clfs)
		if T < 1:
			self.T = self.num_clf
		else:
			self.T = T
	
		self.clfs_picked = [] # list of classifiers h_t for t=0,...,T-1
		self.betas = []       # list of weights beta_t for t=0,...,T-1
		return

	@abstractmethod
	def train(self, features: List[List[float]], labels: List[int]):
		return

	def predict(self, features: List[List[float]]) -> List[int]:
		N = len(features)
		y_sum = np.zeros(N)
		for t in range(0,self.T): # aggrate classifiers
			y_t = self.clfs_picked[t].predict(features)
			y_t = np.array(y_t)
			y_sum += self.betas[t]*y_t
		labels = []
		for y in y_sum: # get labels
			if y>0:
				labels.append(1)
			else:
				labels.append(-1)
		assert len(labels) == N
		return labels
		

class AdaBoost(Boosting):
	def __init__(self, clfs: Set[Classifier], T=0):
		Boosting.__init__(self, clfs, T)
		self.clf_name = "AdaBoost"
		return
		
	def train(self, features: List[List[float]], labels: List[int]):
		N = len(labels)
		labels =  np.array(labels)
		#initialization w_0
		w = np.array([1/N]*N)
		indicator = np.zeros(N)
		# print(features)
		# print(labels)
		for t in range(0,self.T):
			#find h_t and compute eta_t
			eta_t = 1
			for clf in self.clfs:
				labels_clf = np.array(clf.predict(features))
				# print(labels_clf)
				indicator_clf = np.heaviside(np.negative(np.multiply(labels,labels_clf)),0)
				# print(indicator_clf)
				error = np.dot(w,indicator_clf.T)
				if error < eta_t:
					eta_t =  error
					h_t = clf
					indicator = indicator_clf
			#compute beta_t
			beta_t =  0.5*np.log((1-eta_t)/eta_t)
			#compute w
			for n in range(0,N):
				if indicator[n] == 0:
					w[n] = w[n]*np.exp(-beta_t)
				else:
					w[n] = w[n]*np.exp(beta_t)
			#normalize w
			w_sum =  np.sum(w)
			for n in range(0,N):
				w[n] = w[n]/w_sum
			#record h_t and beta_t
			self.clfs_picked.append(h_t)
			self.betas.append(beta_t)

		
		
	def predict(self, features: List[List[float]]) -> List[int]:
		return Boosting.predict(self, features)


class LogitBoost(Boosting):
	def __init__(self, clfs: Set[Classifier], T=0):
		Boosting.__init__(self, clfs, T)
		self.clf_name = "LogitBoost"
		return

	def train(self, features: List[List[float]], labels: List[int]):
		N = len(labels)
		labels = np.array(labels)
		f = np.zeros(N)
		# Initialization
		pi =  np.array([0.5]*N)
		h_tX = np.zeros(N)
		for t in range(0,self.T):
			#compute the working response
			z = []
			for n in range(0,N):
				z.append(((labels[n]+1)/2-pi[n])/pi[n]*(1-pi[n]))
			z = np.array(z)
			#compute the weight
			w = []
			for n in range(0,N):
				w.append(pi[n]*(1-pi[n]))
			w = np.array(w)
			#find h_t
			error_min = -1
			for clf in self.clfs:
				labels_clf = np.array(clf.predict(features))
				error = np.dot(w,np.square(z-labels_clf))
				if error <= error_min or error_min == -1:
					error_min = error
					h_t = clf
					h_tX = labels_clf
			#update
			f += 0.5*h_tX
			#compute pi
			for n in range(0,N):
				pi[n] = 1/(1+np.exp(-2*f[n]))
			self.clfs_picked.append(h_t)
			self.betas.append(0.5)
		
		
	def predict(self, features: List[List[float]]) -> List[int]:
		return Boosting.predict(self, features)
	