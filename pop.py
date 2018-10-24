# !/bin/python3
"""
recommendate user with item according to it's popularity
"""
import numpy as np
from leave_one_dataset import LeaveOneDataset
from evaluate import evaluate_model
from time import time
class Pop:
	def __init__(self):
		pass
	def fit(self,ratings):
		self.item_pop = np.sum(ratings,0)
	def predict(self,test,batch_size=100,verbose=0):
		_,items = test
		y_ = []
		for it in items:
			y_.append(self.item_pop[it])
		return y_
# test
if __name__ == '__main__':
	topK = 10
	evaluation_threads = 1
	ds = LeaveOneDataset()
	ds.load('./data/ml100k')
	model = Pop()
	model.fit(ds.train_matrix.toarray())
	testRatings = ds.test_pairs.values[:,:2]
	testNegatives = ds.test_pairs.values[:,2:]
	# Init performance
	t1 = time()
	(hits, ndcgs) = evaluate_model(model, testRatings, testNegatives, topK, evaluation_threads)
	hr, ndcg = np.array(hits).mean(), np.array(ndcgs).mean()
	#mf_embedding_norm = np.linalg.norm(model.get_layer('user_embedding').get_weights())+np.linalg.norm(model.get_layer('item_embedding').get_weights())
	#p_norm = np.linalg.norm(model.get_layer('prediction').get_weights()[0])
	print('Init: HR = %.4f, NDCG = %.4f\t [%.1f s]' % (hr, ndcg, time()-t1))


