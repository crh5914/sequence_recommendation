# !/bin/python3
"""
recommendate user with item according to it's popularity
"""
import numpy as np
from dataset import DataSet
from evaluate import get_hr,get_mrr,get_ndcg
class POP:
	def __init__(self):
		pass
	def fit(self,ratings):
		self.ratings = ratings
		self.item_count = len(ratings[0])
		self.item_pop = np.sum(ratings,0)
	def predict(self,test):
		y_ = []
		for u,tar in test.items():
			proposals = []
			for i,r in enumerate(self.ratings[u]):
				if r <= 0:
					proposals.append((i,self.item_pop[i]))
			ranklist = [ i for i,r in sorted(proposals,key=lambda item: item[1],reverse=True)]
			# ranklist = np.argsort(self.item_pop)[::-1]
			y_.append(ranklist)
		return y_
# test
if __name__ == '__main__':
	ds = DataSet('./data/ml1m.ratings','::')
	# ds = DataSet('./data/ml100k.ratings')
	ds.split()
	ratings = ds.get_implicit_matrix()
	test = ds.get_testdict()
	pop = POP()
	pop.fit(ratings)
	y_ = pop.predict(test)
	y = test.values()
	# print(y)
	k = 10
	hr = get_hr(y_,y,k)
	mrr = get_mrr(y_,y,k)
	ndcg = get_ndcg(y_,y,k)
	print('HR@{}:{},MRR@{}:{},ndcg@{}:{}'.format(k,hr,k,mrr,k,ndcg))


