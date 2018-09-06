# !/bin/python3
"""
Some metrics often used in recommendation literatures.
"""
import math
def get_hr(ranklist,target,k=-1):
	if isinstance(target,int):
		return float(target in ranklist[:k])
	hr = 0
	for ranks,tar in zip(ranklist,target):
		hr += float(tar in ranks[:k])
	return hr/len(target)
def get_mrr(ranklist,target,k=-1):
	if isinstance(target,int):
		ranklist = ranklist[:k]
		for i,item in enumerate(ranklist):
			if item == target:
				return 1/(i+1)
		return 0
	mrr = 0
	for ranks,tar in zip(ranklist,target):
		ranks = ranks[:k]
		for i,item in enumerate(ranks):
			if item == tar:
				mrr += 1/(i+1)
				break
	return mrr/len(target)
def get_ndcg(ranklist,target,k=-1):
	if isinstance(target,int):
		ranklist = ranklist[:k]
		for i,item in enumerate(ranklist):
			if item == target:
				return math.log(2)/math.log(i+2)
		return 0
	ndcg = 0
	for ranks, tar in zip(ranklist,target):
		ranks = ranks[:k]
		for i,item in enumerate(ranks):
			if item == tar:
				ndcg += math.log(2)/math.log(i+2)
				break
	return ndcg/len(target)
# test
if __name__ == '__main__':
	ranklist = [1,2,3,4,5,6,7,8]
	print(get_hr(ranklist,3,5))
	print(get_mrr(ranklist,5,5))
	print(get_ndcg(ranklist,5,5))
	ranklist = [[1,2,2,3,4],[2,3,4,5],[4,6,7,8]]
	target = [2,0,6]
	print(get_hr(ranklist,target))
	print(get_mrr(ranklist,target))
	print(get_ndcg(ranklist,target))





