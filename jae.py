import math
import numpy as np
from functools import reduce

import keras
from keras import Model,backend as K,regularizers
from keras.layers import Dense,Embedding,Input,Reshape,Subtract,Lambda,Flatten,Multiply,Concatenate,Dropout
from keras.optimizers import Adadelta,Adam
import pandas as pd
import time
from scipy import sparse
from time import time
from evaluate import evaluate_model
import sys
import argparse
import multiprocessing as mp
from leave_one_dataset import LeaveOneDataset
#from tqdm import tqdm
def parse_args():
    parser = argparse.ArgumentParser(description="Run MLP.")
    parser.add_argument('--path', nargs='?', default='Data/',
                        help='Input data path.')
    parser.add_argument('--dataset', nargs='?', default='ml-1m',
                        help='Choose a dataset.')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of epochs.')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='Batch size.')
    parser.add_argument('--layers', nargs='?', default='[64,32,16,8]',
                        help="Size of each layer. Note that the first layer is the concatenation of user and item embeddings. So layers[0]/2 is the embedding size.")
    parser.add_argument('--regs', nargs='?', default='0',
                        help="Regularization for each layer")
    parser.add_argument('--num_neg', type=int, default=4,
                        help='Number of negative instances to pair with a positive instance.')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate.')
    parser.add_argument('--learner', nargs='?', default='adam',
                        help='Specify an optimizer: adagrad, adam, rmsprop, sgd')
    parser.add_argument('--verbose', type=int, default=1,
                        help='Show performance per X iterations')
    parser.add_argument('--out', type=int, default=1,
                        help='Whether to save the trained model.')
    return parser.parse_args()

def get_model(num_users,num_items,ratings,layers,l2_param):
	input_a = Input(shape=(1,),name='input-a',dtype='int32')
	input_b = Input(shape=(1,),name='input-b',dtype='int32')
	user_embedding_layer = Embedding(input_dim=num_users,output_dim=num_items,trainable=False,input_length=1,name='user-table')
	user_embedding_layer.build((None,))
	user_embedding_layer.set_weights([ratings])
	user_embedding = user_embedding_layer(input_a)
	user_embedding = Flatten()(user_embedding)
	item_embedding_layer = Embedding(input_dim=num_items,output_dim=num_users,trainable=False,input_length=1,name='item-table')
	item_embedding_layer.build((None,))
	item_embedding_layer.set_weights([ratings.T])
	item_embedding = item_embedding_layer(input_b)
	item_embedding = Flatten()(item_embedding)
	user_hidden = Dense(layers[0],activation='relu',kernel_regularizer=regularizers.l2(l2_param),name='user-encoding-layer-{}'.format(0))(user_embedding)
	item_hidden = Dense(layers[0],activation='relu',kernel_regularizer=regularizers.l2(l2_param),name='item-encoding-layer-{}'.format(0))(item_embedding)
	encoded_a = Dense(layers[1],activation='relu',kernel_regularizer=regularizers.l1(l2_param),name='user-encoding-layer-{}'.format(1))(user_hidden)
	encoded_b = Dense(layers[1],activation='relu',kernel_regularizer=regularizers.l1(l2_param),name='item-encoding-layer-{}'.format(1))(item_hidden)
	user_out_hidden = Dense(layers[0],activation='relu',kernel_regularizer=regularizers.l2(l2_param),name='user-decoding-layer-{}'.format(0))(encoded_a)
	item_out_hidden = Dense(layers[0],activation='relu',kernel_regularizer=regularizers.l2(l2_param),name='item-decoding-layer-{}'.format(0))(encoded_b)
	decoded_a = Dense(num_items,activation='sigmoid',kernel_regularizer=regularizers.l2(l2_param),name='user-decoding-layer-{}'.format(1))(user_out_hidden)
	decoded_b = Dense(num_users,activation='sigmoid',kernel_regularizer=regularizers.l2(l2_param),name='item-decoding-layer-{}'.format(1))(item_out_hidden)
	embedding_diff = Concatenate()([encoded_a,encoded_b])
	result_hidden = Dense(layers[1],activation='relu',kernel_regularizer=regularizers.l2(l2_param),name='predict-layer-{}'.format(0))(embedding_diff)
	result =  Dense(1,activation='sigmoid',kernel_regularizer=regularizers.l2(l2_param),name='predict-layer-{}'.format(1))(result_hidden)
	model = Model(inputs=[input_a,input_b],outputs=[decoded_a,decoded_b,result])
	# opt = Adam(lr=1e-4)
	# model.compile(optimizer=opt,loss=[recostruction_loss,recostruction_loss,edge_wise_loss],loss_weights=[1,1,alpha])
	predictor = Model(inputs=[input_a,input_b],outputs=[result])
	# predictor.compile(optimizer=opt,loss=[edge_wise_loss],metrics=[edge_wise_loss])
	# model.summary()
	encoder = Model(input_a,encoded_a)
	decoder = Model(input_a,decoded_a)
	# decoder.compile(optimizer='adadelta',loss=recostruction_loss)
	return model,encoder,decoder,predictor

def recostruction_loss(true_y,pred_y):
    diff = K.square(true_y - pred_y)
    weight =K.cast(true_y>0,dtype='float32')
    weighted_diff = diff * weight
    return K.mean(K.sum(weighted_diff,axis=1))
def edge_wise_loss(true_y,pred_y):
    """ 1st order proximity
    """
    return K.mean(K.square(pred_y-true_y))

def get_train_instances(train_pairs,ratings,batch_size=512):
	user_input, item_input,user_vec,item_vec,labels = [],[],[],[],[]
	train_pairs = train_pairs.values
	count = 0
	for pair in train_pairs:
		# positive instance
		u,i = pair[0],pair[1]
		user_input.append(u)
		item_input.append(i)
		user_vec.append(ratings[u])
		item_vec.append(ratings.T[i])
		labels.append(1)
		count += 1
		# negative instances
		for j in pair[2:]:
			user_input.append(u)
			item_input.append(j)
			user_vec.append(ratings[u])
			item_vec.append(ratings.T[i])
			labels.append(0)
			count += 1
		if count >= batch_size:
			count = 0
			yield [user_input,item_input,user_vec,item_vec,labels]
			user_input, item_input,user_vec,item_vec,labels = [],[],[],[],[]
	if count > 0:
		count = 0
		yield [user_input,item_input,user_vec,item_vec,labels]
if __name__ == '__main__':
	args = parse_args()
	path = args.path
	dataset = args.dataset
	layers = eval(args.layers)
	regs = eval(args.regs)
	num_negatives = args.num_neg
	learner = args.learner
	learning_rate = args.lr
	batch_size = args.batch_size
	epochs = args.epochs
	verbose = args.verbose

	topK = 10
	evaluation_threads = 1 #mp.cpu_count()
	print("jae arguments: %s " %(args))
	model_out_file = 'result/%s_jae_%s_%d.h5' %(args.dataset, args.layers, time())

	# Loading data
	t1 = time()
	# dataset = Dataset(args.path + args.dataset)
	ds = LeaveOneDataset()
	ds.load('./data/%s'%args.dataset)
	ratings = ds.train_matrix.toarray()
	testRatings = ds.test_pairs.values[:,:2]
	testNegatives = ds.test_pairs.values[:,2:]
	num_users, num_items = ds.num_users,ds.num_items
	print("Load data done [%.1f s]. #user=%d, #item=%d, #train=%d, #test=%d" 
			%(time()-t1, num_users, num_items, len(ds.train_pairs), len(testRatings)))
    
    # Build model
	model,encoder,decoder,predictor = get_model(num_users, num_items,ratings,layers, regs)
	if learner.lower() == "adagrad": 
		model.compile(optimizer=Adagrad(lr=learning_rate),loss=['kullback_leibler_divergence','kullback_leibler_divergence','binary_crossentropy'])
	elif learner.lower() == "rmsprop":
		model.compile(optimizer=RMSprop(lr=learning_rate),loss=[recostruction_loss,recostruction_loss,'binary_crossentropy'])
	elif learner.lower() == "adam":
		model.compile(optimizer=Adam(lr=learning_rate),loss=[recostruction_loss,recostruction_loss,'binary_crossentropy'])
	else:
		model.compile(optimizer=SGD(lr=learning_rate),loss=[recostruction_loss,recostruction_loss,'binary_crossentropy'])    
    
    # Check Init performance
	t1 = time()
	(hits, ndcgs) = evaluate_model(predictor, testRatings, testNegatives, topK, evaluation_threads)
	hr, ndcg = np.array(hits).mean(), np.array(ndcgs).mean()
	print('Init: HR = %.4f, NDCG = %.4f [%.1f]' %(hr, ndcg, time()-t1))
    
    # Train model
	best_hr, best_ndcg, best_iter = hr, ndcg, -1
	for epoch in range(epochs):
		t1 = time()
		loss = 0
		count = 0
        # Generate training instances
		for user_input, item_input,user_vec,item_vec,labels in get_train_instances(ds.train_pairs,ratings,batch_size=256):
			# Training      
			_,_,_,hist = model.train_on_batch([np.array(user_input), np.array(item_input)],[np.array(user_vec),np.array(item_vec),np.array(labels)])
			count += 1
			loss += hist
		loss = loss/count
		t2 = time()
		# Evaluation
		if epoch %verbose == 0:
			(hits, ndcgs) = evaluate_model(predictor, testRatings, testNegatives, topK, evaluation_threads)
			hr, ndcg, loss = np.array(hits).mean(), np.array(ndcgs).mean(), loss
			print('Iteration %d [%.1f s]: HR = %.4f, NDCG = %.4f, loss = %.4f [%.1f s]' 
			% (epoch,  t2-t1, hr, ndcg, loss, time()-t2))
			if hr > best_hr:
				best_hr, best_ndcg, best_iter = hr, ndcg, epoch
				if args.out > 0:
					model.save_weights(model_out_file, overwrite=True)

	print("End. Best Iteration %d:  HR = %.4f, NDCG = %.4f. " %(best_iter, best_hr, best_ndcg))
	if args.out > 0:
		print("The best JAE model is saved to %s" %(model_out_file))

 
