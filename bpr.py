import numpy as np

# import theano
# import theano.tensor as T
import keras
from keras import backend as K
from keras import initializers
from keras.regularizers import l2
from keras.models import Sequential,Model
from keras.layers.core import Dense, Lambda, Activation
from keras.layers import Embedding, Input, Dense, Subtract,Dot, Reshape, Flatten, Dropout
from keras.constraints import maxnorm
from keras.optimizers import Adagrad, Adam, SGD, RMSprop
from evaluate import evaluate_model
# from Dataset import Dataset
from time import time
import sys
import argparse
import multiprocessing as mp
from leave_one_dataset import LeaveOneDataset

def parse_args():
    parser = argparse.ArgumentParser(description="Run BPR.")
    parser.add_argument('--dataset', nargs='?', default='ml1m',
                        help='Choose a dataset.')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of epochs.')
    parser.add_argument('--num_factors', type=int, default=8,
                        help='Embedding size of MF model.')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='Batch size.')
    parser.add_argument('--regs', nargs='?', default='[0,0]',
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

def get_model(num_users, num_items,latent_dim,regs=[0,0]):
    # Input variables
    user_input = Input(shape=(1,), dtype='int32', name = 'user_input')
    item_input = Input(shape=(1,), dtype='int32', name = 'item_input')
    neg_item_input = Input(shape=(1,), dtype='int32', name = 'neg_item_input')


    MF_Embedding_User = Embedding(input_dim = num_users, output_dim = latent_dim, name = 'user_embedding',
                                  embeddings_initializer = initializers.RandomNormal(mean=0.0, stddev=0.01, seed=None), embeddings_regularizer = l2(regs[0]), input_length=1)
    MF_Embedding_Item = Embedding(input_dim = num_items, output_dim = latent_dim, name = 'item_embedding',
                                  embeddings_initializer = initializers.RandomNormal(mean=0.0, stddev=0.01, seed=None), embeddings_regularizer = l2(regs[1]), input_length=1)   
    
    # Crucial to flatten an embedding vector!
    user_latent = Flatten()(MF_Embedding_User(user_input))
    item_latent = Flatten()(MF_Embedding_Item(item_input))
    neg_latent = Flatten()(MF_Embedding_Item(neg_item_input))
    # The 0-th layer is the concatenation of embedding layers
    delta_latent = Subtract()([item_latent,neg_latent])
    y = Dot(axes=1)([user_latent,item_latent])
    prediction = Dot(axes=1)([user_latent,delta_latent])
    # Final prediction layer
    prediction = Activation('sigmoid')(prediction)
    
    train_model = Model(input=[user_input, item_input,neg_item_input], 
                  output=prediction)
    predict_model = Model(input=[user_input, item_input], 
                  output=y)
    return train_model,predict_model

def get_train_instances(train_pairs):
    user_input, item_input,neg_item_input,labels = [],[],[],[]
    train_pairs = train_pairs.values
    for pair in train_pairs:
        # positive instance
        u,i = pair[0],pair[1]
        # negative instances
        for j in pair[2:]:
            user_input.append(u)
            item_input.append(i)
            neg_item_input.append(j)
            labels.append(1)
    return user_input, item_input,neg_item_input,labels

if __name__ == '__main__':
    args = parse_args()
    # path = args.path
    dataset = args.dataset
    regs = eval(args.regs)
    num_negatives = args.num_neg
    num_factors = args.num_factors
    learner = args.learner
    learning_rate = args.lr
    batch_size = args.batch_size
    epochs = args.epochs
    verbose = args.verbose
    
    topK = 10
    evaluation_threads = 1 #mp.cpu_count()
    print("BPR arguments: %s " %(args))
    model_out_file = 'result/%s_bpr_%d_%d.h5' %(args.dataset,args.num_factors,time())
    
    # Loading data
    t1 = time()
    # dataset = Dataset(args.path + args.dataset)
    ds = LeaveOneDataset()
    ds.load('./data/%s'%args.dataset)
    # train, testRatings, testNegatives = dataset.trainMatrix, dataset.testRatings, dataset.testNegatives
    testRatings = ds.test_pairs.values[:,:2]
    testNegatives = ds.test_pairs.values[:,2:]
    num_users, num_items = ds.num_users,ds.num_items
    print("Load data done [%.1f s]. #user=%d, #item=%d, #train=%d, #test=%d" 
          %(time()-t1, num_users, num_items, len(ds.train_pairs), len(testRatings)))
    
    # Build model
    train_model,predict_model = get_model(num_users, num_items,num_factors,regs)
    if learner.lower() == "adagrad": 
        train_model.compile(optimizer=Adagrad(lr=learning_rate), loss='binary_crossentropy')
    elif learner.lower() == "rmsprop":
        train_model.compile(optimizer=RMSprop(lr=learning_rate), loss='binary_crossentropy')
    elif learner.lower() == "adam":
        train_model.compile(optimizer=Adam(lr=learning_rate), loss='binary_crossentropy')
    else:
        train_model.compile(optimizer=SGD(lr=learning_rate), loss='binary_crossentropy')    
    
    # Check Init performance
    t1 = time()
    (hits, ndcgs) = evaluate_model(predict_model, testRatings, testNegatives, topK, evaluation_threads)
    hr, ndcg = np.array(hits).mean(), np.array(ndcgs).mean()
    print('Init: HR = %.4f, NDCG = %.4f [%.1f]' %(hr, ndcg, time()-t1))
    
    # Train model
    best_hr, best_ndcg, best_iter = hr, ndcg, -1
    for epoch in range(epochs):
        t1 = time()
        # Generate training instances
        user_input, item_input,neg_item_input,labels = get_train_instances(ds.train_pairs)
    
        # Training        
        hist = train_model.fit([np.array(user_input), np.array(item_input), np.array(neg_item_input)],#input
                         np.array(labels), # labels 
                         batch_size=batch_size, epochs=1, verbose=0, shuffle=True)
        t2 = time()

        # Evaluation
        if epoch %verbose == 0:
            (hits, ndcgs) = evaluate_model(predict_model, testRatings, testNegatives, topK, evaluation_threads)
            hr, ndcg, loss = np.array(hits).mean(), np.array(ndcgs).mean(), hist.history['loss'][0]
            print('Iteration %d [%.1f s]: HR = %.4f, NDCG = %.4f, loss = %.4f [%.1f s]' 
                  % (epoch,  t2-t1, hr, ndcg, loss, time()-t2))
            if hr > best_hr:
                best_hr, best_ndcg, best_iter = hr, ndcg, epoch
                if args.out > 0:
                    train_model.save_weights(model_out_file, overwrite=True)

    print("End. Best Iteration %d:  HR = %.4f, NDCG = %.4f. " %(best_iter, best_hr, best_ndcg))
    if args.out > 0:
        print("The best BPR model is saved to %s" %(model_out_file))