import tensorflow as tf
import numpy as np
from leave_one_dataset import LeaveOneDataset
from time import time
from evaluate import getHitRatio,getNDCG
import argparse
from dataset import DataSet
def parse_args():
    parser = argparse.ArgumentParser(description="Run NeuMF.")
    parser.add_argument('--path', nargs='?', default='Data/',
                        help='Input data path.')
    parser.add_argument('--dataset', nargs='?', default='ml100k',
                        help='Choose a dataset.')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of epochs.')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='Batch size.')
    parser.add_argument('--num_factors', type=int, default=32,
                        help='Embedding size of MF model.')
    # parser.add_argument('--layers', nargs='?', default='[64,32,16,8]',
    #                     help="MLP layers. Note that the first layer is the concatenation of user and item embeddings. So layers[0]/2 is the embedding size.")
    # parser.add_argument('--reg_mf', type=float, default=0,
    #                     help='Regularization for MF embeddings.')
    parser.add_argument('--reg_lambda', type=float, default=0.01,
                        help="regurizeration parameter")
    parser.add_argument('--num_neg', type=int, default=4,
                        help='Number of negative instances to pair with a positive instance.')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate.')
    parser.add_argument('--keep_prob', type=float, default=0.8,
                        help='keep probability')
    parser.add_argument('--learner', nargs='?', default='adam',
                        help='Specify an optimizer: adagrad, adam, rmsprop, sgd')
    parser.add_argument('--verbose', type=int, default=1,
                        help='Show performance per X iterations')
    parser.add_argument('--out', type=int, default=0,
                        help='Whether to save the trained model.')
    return parser.parse_args()
class TwoLevelAttetionModel:
    def __init__(self,sess,num_users,num_items,num_factors,max_len,lr,reg_lambda,keep_prob):
        self.sess = sess
        self.num_users = num_users
        self.num_items = num_items
        self.max_len = max_len
        self.num_factors = num_factors
        self.reg_lambda = reg_lambda
        self.lr = lr
        self.keep_prob = keep_prob
        self.build_model()
        # self.epochs = epochs
    def build_model(self):
        self.user = tf.placeholder(shape=[None],dtype=tf.int32)
        self.item = tf.placeholder(shape=[None],dtype=tf.int32)
        self.backets = tf.placeholder(shape=[None,self.max_len],dtype=tf.int32)
        self.mask = tf.placeholder(shape=[None], dtype=tf.int32)
        self.y = tf.placeholder(shape=[None],dtype=tf.float32)
        self.dropout = tf.placeholder(dtype=tf.float32)
        self.item_embedding = tf.Variable(tf.random_uniform(shape=(self.num_items,self.num_factors),minval=-0.1,maxval=0.1))
        self.user_embedding = tf.Variable(tf.random_uniform(shape=(self.num_users,self.num_factors),minval=-0.1,maxval=0.1))
        #self.fa_W = tf.Variable(tf.random_normal(shape=(2*self.num_factors,self.num_factors)))
        #self.fa_b = tf.Variable(tf.constant(0.1,shape=(self.num_factors,)))
        #self.it_W = tf.Variable(tf.random_normal(shape=(2*self.num_factors,self.max_len)))
        #self.it_b = tf.Variable(tf.constant(0.1,shape=[self.max_len]))
        #self.W1 = tf.Variable(tf.random_normal(shape=(3*self.num_factors,int(1.5*self.num_factors))))
        #self.b1 = tf.Variable(tf.constant(0.1,shape=(int(1.5*self.num_factors),)))
        #self.W2 = tf.Variable(tf.random_normal(shape=(int(1.5*self.num_factors),1)))
        #self.b2 = tf.Variable(tf.constant(0.1))
        self.mask_vec = tf.expand_dims(tf.cast(tf.sequence_mask(self.mask,self.max_len),dtype=tf.float32),axis=-1)
        self.backets_embedding = tf.nn.embedding_lookup(self.item_embedding,self.backets)
        self.backets_embedding = tf.multiply(self.mask_vec,self.backets_embedding)
        self.item_vec = tf.nn.embedding_lookup(self.item_embedding,self.item)
        self.user_vec = tf.nn.embedding_lookup(self.user_embedding,self.user)
        self.factor_attented_backets_vec = self.factor_attention(self.user_vec,self.item_vec,self.backets_embedding)
        self.full_attented_backet_vec = self.backets_attention(self.user_vec,self.item_vec,self.factor_attented_backets_vec)
        self.final_vec = tf.concat([self.item_vec,self.user_vec,self.full_attented_backet_vec],axis=1)
        #self.f1 = tf.nn.relu(tf.add(tf.matmul(self.final_vec,self.W1),self.b1))
        #self.f1 = tf.nn.dropout(self.f1,self.dropout)
        #self.y_ = tf.nn.sigmoid(tf.reduce_sum(tf.add(tf.matmul(self.f1,self.W2),self.b2),axis=1))
        self.f1 = tf.layers.dense(self.final_vec,int(1.5*self.num_factors),activation=tf.nn.relu,name='full_layer_1')
        self.f2 = tf.layers.dense(self.f1,1,activation=tf.nn.sigmoid,name='full_layer_2')
        self.y_ = tf.reduce_sum(self.f2,axis=1)
        self.log_loss = tf.losses.log_loss(self.y,self.y_)
        #self.l2_loss = tf.nn.l2_loss(self.W1)
        #self.l2_loss += tf.nn.l2_loss(self.b1)
        #self.l2_loss += tf.nn.l2_loss(self.W2)
        #self.l2_loss += tf.nn.l2_loss(self.b2)
        #self.loss = self.log_loss + self.reg_lambda * self.l2_loss
        self.loss = self.log_loss
        self.train_opt = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.loss)
        init = tf.global_variables_initializer()
        self.sess.run(init)
    def factor_attention(self,user_vec,item_vec,backets_vec):
        fusion_vec = tf.concat([user_vec, item_vec], axis=-1)
        factor_attetion_hidden = tf.layers.dense(fusion_vec,self.num_factors,activation=tf.nn.relu,name='factor_attention_hidden_layer')
        #alphas = tf.nn.softmax(tf.add(tf.matmul(fusion_vec, W), b))
        alphas = tf.layers.dense(factor_attetion_hidden,self.num_factors,activation=tf.nn.softmax,name='factor_attention_layer')
        alphas = tf.expand_dims(alphas,axis=1)
        attented_vec = tf.multiply(alphas,backets_vec)
        return attented_vec
    def backets_attention(self,user_vec,item_vec,backets_vec):
        fusion_vec = tf.concat([user_vec, item_vec], axis=-1)
        backet_attetion_hidden = tf.layers.dense(fusion_vec,self.num_factors,activation=tf.nn.relu,name='backet_attention_hidden_layer')
        betas = tf.layers.dense(backet_attetion_hidden,self.max_len,activation=tf.nn.softmax,name='backet_attention_layer')
        #betas = tf.nn.softmax(tf.add(tf.matmul(fusion_vec, W), b))
        item_attention_weights = tf.expand_dims(betas, axis=-1)
        aggregated_backet_vec = tf.reduce_sum(tf.multiply(item_attention_weights, backets_vec), axis=1)
        return aggregated_backet_vec
    def train(self,batch_users,batch_items,batch_uvecs,batch_masks,batch_labels):
        feed_dict = {self.user:batch_users,self.item:batch_items,self.backets:batch_uvecs,self.mask:batch_masks,self.y:batch_labels,self.dropout:self.keep_prob}
        _,loss,y_ = self.sess.run([self.train_opt,self.loss,self.y_],feed_dict=feed_dict)
        return loss
    def predict(self,batch_users, batch_items, batch_uvecs, batch_masks):
        feed_dict = {self.user:batch_users,self.item:batch_items,self.backets:batch_uvecs,self.mask:batch_masks,self.dropout:1.0}
        y_ = self.sess.run(self.y_,feed_dict=feed_dict)
        return y_
def get_train_batch(train_pairs,train_matrix,user_len,batch_size=256):
    batch_users,batch_items,batch_uvecs,batch_masks,batch_labels = [],[],[],[],[]
    count = 0
    for pair in train_pairs:
        u,i = pair[0],pair[1]
        uvec = list(np.nonzero(train_matrix[u])[0])
        padd_len = user_len - len(uvec)
        padd_uvec = uvec + [0]*padd_len
        batch_users.append(u)
        batch_uvecs.append(padd_uvec)
        batch_items.append(i)
        batch_masks.append(len(uvec))
        batch_labels.append(1)
        count += 1
        for j in pair[2:]:
            batch_users.append(u)
            batch_uvecs.append(padd_uvec)
            batch_items.append(j)
            batch_masks.append(len(uvec))
            batch_labels.append(0)
            count += 1
        if count >= batch_size:
            yield batch_users,batch_items,batch_uvecs,batch_masks,batch_labels
            batch_users, batch_items, batch_uvecs, batch_masks, batch_labels = [], [], [], [], []
            count = 0
    if count >= 0:
        yield batch_users, batch_items, batch_uvecs, batch_masks, batch_labels
def generate_train_batch(users,items,train_matrix,user_len,item_count,batch_size=128,negative=2):
    batch_users,batch_items,batch_uvecs,batch_masks,batch_labels = [],[],[],[],[]
    count = 0
    for u,i in zip(users,items):
        uvec = list(np.nonzero(train_matrix[u])[0])
        padd_len = user_len - len(uvec)
        padd_uvec = uvec + [0]*padd_len
        batch_users.append(u)
        batch_uvecs.append(padd_uvec)
        batch_items.append(i)
        batch_masks.append(len(uvec))
        batch_labels.append(1)
        count += 1
        num = 0
        while num < negative:
            j = np.random.choice(item_count)
            if j not in uvec:
                batch_users.append(u)
                batch_uvecs.append(padd_uvec)
                batch_items.append(j)
                batch_masks.append(len(uvec))
                batch_labels.append(0)
                num += 1
                count += 1
        if count >= batch_size:
            yield batch_users,batch_items,batch_uvecs,batch_masks,batch_labels
            batch_users, batch_items, batch_uvecs, batch_masks, batch_labels = [], [], [], [], []
            count = 0
    if count >= 0:
        yield batch_users, batch_items, batch_uvecs, batch_masks, batch_labels
def get_test_batch(test_pair,train_matrix,user_len):
    for pair in test_pair:
        batch_users, batch_items, batch_uvecs, batch_masks = [], [], [], []
        u,i = pair[0],pair[1]
        uvec = list(np.nonzero(train_matrix[u])[0])
        padd_len = user_len - len(uvec)
        padd_uvec = uvec + [0]*padd_len
        its = list(set(range(num_items))-set(uvec))
        batch_users.append(u)
        batch_uvecs.append(padd_uvec)
        batch_masks.append(len(uvec))
        batch_items.append(i)
        for j in pair[2:]:
            batch_users.append(u)
            batch_uvecs.append(padd_uvec)
            batch_masks.append(len(uvec))
            batch_items.append(j)
        yield batch_users, batch_items, batch_uvecs, batch_masks
def generate_test_batch(test,train_matrix,user_len,num_items):
    for u in test:
        batch_users, batch_items, batch_uvecs, batch_masks = [], [], [], []
        i = test[u]
        uvec = list(np.nonzero(train_matrix[u])[0])
        padd_len = user_len - len(uvec)
        padd_uvec = uvec + [0]*padd_len
        its = list(set(range(num_items))-set(uvec))
        batch_users.append(u)
        batch_uvecs.append(padd_uvec)
        batch_masks.append(len(uvec))
        batch_items.append(i)
        negs = np.random.choice(its,size=99)
        for j in negs:
            batch_users.append(u)
            batch_uvecs.append(padd_uvec)
            batch_masks.append(len(uvec))
            batch_items.append(j)
        yield batch_users, batch_items, batch_uvecs, batch_masks
if __name__ == '__main__':
    ds = LeaveOneDataset()
    #ds = DataSet('./data/ml100k.ratings')
    #ds.split()
    ds.load('./data/ml100k')
    train_matrix = ds.train_matrix.toarray()
    #train_matrix = ds.get_implicit_matrix()
    #test = ds.get_testdict()
    test = ds.test_pairs.values
    user_len = np.max(np.sum(train_matrix>0,axis=1))
    #num_users = ds.user_count
    num_users = ds.num_users
    num_items = ds.num_items
    #num_items = ds.item_count
    # print(user_len)
    args = parse_args()
    topK = 10
    sess = tf.Session()
    model = TwoLevelAttetionModel(sess,num_users,num_items,args.num_factors,user_len,args.lr,args.reg_lambda,args.keep_prob)
    init_hits,init_ndcgs = [],[]
    start = time()
    for batch_users, batch_items, batch_uvecs, batch_masks in get_test_batch(test,train_matrix,user_len):
        scores = model.predict(batch_users, batch_items, batch_uvecs, batch_masks)
        scores = np.reshape(scores, -1)
        ranklist = np.argsort(-scores)[:topK]
        init_hits.append(getHitRatio(ranklist, 0))
        init_ndcgs.append(getNDCG(ranklist, 0))
    init_hit = np.mean(init_hits)
    init_ndcg = np.mean(init_ndcgs)
    print('Init,hit@{}:{},ndcg@{}:{},{}s'.format(topK, init_hit, topK, init_ndcg,time()-start))
    best_hit = 0
    best_ndcg = 0
    for epoch in range(args.epochs):
        start = time()
        #train
        for batch_users, batch_items, batch_uvecs, batch_masks, batch_labels in get_train_batch(ds.train_pairs.values,train_matrix,user_len):
            loss = model.train(batch_users,batch_items,batch_uvecs, batch_masks, batch_labels)
        # test
        hits = []
        ndcgs = []
        for batch_users, batch_items, batch_uvecs, batch_masks in get_test_batch(test,train_matrix,user_len):
            scores = model.predict(batch_users,batch_items, batch_uvecs, batch_masks)
            scores = np.reshape(scores,-1)
            ranklist = np.argsort(-scores)[:topK]
            hits.append(getHitRatio(ranklist,0))
            ndcgs.append(getNDCG(ranklist,0))
        hit = np.mean(hits)
        ndcg = np.mean(ndcgs)
        print('epoch:{},loss:{},hit@{}:{},ndcg@{}:{},{}s'.format(epoch,loss,topK,hit,topK,ndcg,time()-start))
        if hit > best_hit:
            best_hit = hit
        if ndcg > best_ndcg:
            best_ndcg = ndcg
    print('best hit@{}:{},best ndcg@{}:{}'.format(topK,best_hit,topK,best_ndcg))




