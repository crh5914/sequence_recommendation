# !/bin/python3
import numpy as np
from collections import defaultdict
from scipy.sparse import csr_matrix
class DataSet:
	def __init__(self,file,sep='\t'):
		self.load(file,sep)
	def load(self,file,sep):
		self.user_map = defaultdict()
		self.item_map = defaultdict()
		self.user_count = 0
		self.item_count = 0
		self.users = []
		self.items = []
		self.ratings = []
		self.ui = defaultdict(list)
		self.iu = defaultdict(list)
		with open(file,'r') as f:
			for no,line in enumerate(f):
				cols = line.strip().split(sep)
				u,i,r,t= int(cols[0]),int(cols[1]),float(cols[2]),int(cols[3])
				if u not in self.user_map:
					self.user_map[u] = self.user_count
					self.user_count += 1
				if i not in self.item_map:
					self.item_map[i] = self.item_count
					self.item_count += 1
				self.ui[self.user_map[u]].append((t,self.item_map[i],r,no))
				self.iu[self.item_map[i]].append((self.user_map[u],r))
				self.users.append(self.user_map[u])
				self.items.append(self.item_map[i])
				self.ratings.append(r)
	def split(self):
		self.test = {}
		self.test_idxs = []
		for u in self.ui:
			self.ui[u] = sorted(self.ui[u],key=lambda item: item[0])
			t,i,r,no = self.ui[u][-1]
			self.test[u] = i
			self.test_idxs.append(no)
	def get_implicit_matrix(self):
		"""
		return a matrix represent implicit feedback
		"""
		data = np.ones(len(self.ratings))
		data[self.test_idxs] = 0
		return csr_matrix((data,(self.users,self.items)),shape=(self.user_count,self.item_count)).toarray()
	def get_explicit_matrix(self):
		"""
		return a matrix represent explicit feedback
		"""
		data = np.array(self.ratings)
		data[self.test_idxs] = 0
		return csr_matrix((data,(self.users,self.items)),shape=(self.user_count,self.item_count)).toarray()
	def get_testdict(self):
		return self.test



