import numpy as np
from typing import List
from classifier import Classifier

class DecisionTree(Classifier):
	def __init__(self):
		self.clf_name = "DecisionTree"
		self.root_node = None

	def train(self, features: List[List[float]], labels: List[int]):
		# init.
		assert(len(features) > 0)
		self.feautre_dim = len(features[0])
		num_cls = np.max(labels)+1
		# print(features)
		# print(labels)
		# build the tree
		self.root_node = TreeNode(features, labels, num_cls)
		if self.root_node.splittable:
			self.root_node.split()

		return
		
	def predict(self, features: List[List[float]]) -> List[int]:
		y_pred = []
		for feature in features:
			y_pred.append(self.root_node.predict(feature))
		return y_pred

	def print_tree(self, node=None, name='node 0', indent=''):
		if node is None:
			node = self.root_node
		print(name + '{')
		if node.splittable:
			print(indent + '  split by dim {:d}'.format(node.dim_split))
			for idx_child, child in enumerate(node.children):
				self.print_tree(node=child, name= '  '+name+'/'+str(idx_child), indent=indent+'  ')
		else:
			print(indent + '  cls', node.cls_max)
		print(indent+'}')


class TreeNode(object):
	def __init__(self, features: List[List[float]], labels: List[int], num_cls: int):
		self.features = features
		self.labels = labels
		self.children = []
		self.num_cls = num_cls

		count_max = 0
		for label in np.unique(labels):
			if self.labels.count(label) > count_max:
				count_max = labels.count(label)
				self.cls_max = label # majority of current node

		if len(np.unique(labels)) < 2:
			self.splittable = False
		else:
			self.splittable = True

		self.dim_split = None # the dim of feature to be splitted

		self.feature_uniq_split = None # the feature to be splitted


	def split(self):
		def conditional_entropy(branches: List[List[int]]) -> float:
			'''
			branches: C x B array, 
					  C is the number of classes,
					  B is the number of branches
					  it stores the number of 
					  corresponding training samples
			'''
			########################################################
			# TODO: compute the conditional entropy
			########################################################
			branches = np.array(branches)
			C,B = branches.shape
			class_sum = np.sum(branches, axis=0) # B dimension array
			probability_bc = np.empty_like(branches)
			probability_b = class_sum/np.sum(branches)
			con_entropy = 0
			for b in range(0,B):
				for c in range(0,C):
					probability_bc[c][b] = branches[c][b]/class_sum[b]
				entropy = 0
				for c in range(0,C):
					if (probability_bc[c][b] != 0):
						entropy -= probability_bc[c][b]*np.log(probability_bc[c][b])
				con_entropy += probability_b[b]*entropy
			return con_entropy

		con_entropy_min = -1
		features = np.array(self.features)
		classes = np.unique(self.labels)
		C = len(classes)
		for idx_dim in range(len(self.features[0])):
		############################################################
		# TODO: compare each split using conditional entropy
		#       find the 
		############################################################
			#get branches metrix
			attribute_values = np.unique(features[:,idx_dim])
			B = len(attribute_values)
			branches = np.zeros((C,B))
			for n in range(0,len(self.labels)):
				for b in range(0,B):
					for c in range(0,C):
						if self.features[n][idx_dim] == attribute_values[b] and self.labels[n] == classes[c]:
							branches[c][b] += 1
			branches.tolist()
			# print("dim =",idx_dim)
			con_entropy = conditional_entropy(branches)
			# print(con_entropy)
			if con_entropy_min == -1 or con_entropy < con_entropy_min:
				self.dim_split = idx_dim
				self.feature_uniq_split = attribute_values
		self.feature_uniq_split = self.feature_uniq_split.tolist()
		############################################################
		# TODO: split the node, add child nodes
		############################################################
		features_removed = np.delete(features,self.dim_split,1)
		features_removed = features_removed.tolist()
		for feature in self.feature_uniq_split:
			child_features =[]
			child_labels = []

			#get instances
			for n in range(0,len(self.labels)):
				if feature == self.features[n][self.dim_split]:
					# print(n)
					child_features.append(features_removed[n])
					child_labels.append(self.labels[n])
			child_num_cls = np.max(child_labels)+1
			child_node = TreeNode(child_features, child_labels, child_num_cls)
			if child_features[0] == []:
				child_node.splittable = False
			self.children.append(child_node)


		# split the child nodes
		for child in self.children:
			if child.splittable:
				child.split()

		return

	def predict(self, feature: List[int]) -> int:
		if self.splittable:
			# print(feature)
			idx_child = self.feature_uniq_split.index(feature[self.dim_split])
			return self.children[idx_child].predict(feature)
		else:
			return self.cls_max



