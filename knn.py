import numpy as np
import pandas as pd
from evaluate import evaluate_model
from time import time
from leave_one_dataset import LeaveOneDataset
class ItemKNN:
    def __init__(self,k):
        self.k = k
    def fit(self,ratings):
        self.ratings = ratings
        cos_similarity = np.dot(ratings.T,ratings)
        item_norm = np.sqrt(np.sum(np.square(ratings),axis=0))
        norm_matrix = np.dot(item_norm.reshape(-1,1),item_norm.reshape(1,-1))
        norm_matrix = np.where(norm_matrix>0,norm_matrix,10**6)
        self.cos_similarity = cos_similarity/norm_matrix
    def predict(self,test,batch_size=256,verbose=0):
        users,items = test
        y_ = []
        for uid,it in zip(users,items):
            us = self.ratings[uid]
            idx = np.argsort(self.cos_similarity[it])[-self.k:]
            score = np.dot(us[idx],self.cos_similarity[it][idx])
            y_.append(score)
        return y_
if __name__ == '__main__':
	topK = 10
	evaluation_threads = 1
	ds = LeaveOneDataset()
	ds.load('./data/ml100k')
	model = ItemKNN(100)
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


