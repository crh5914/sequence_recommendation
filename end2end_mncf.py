import tensorflow as tf
import numpy as np
from Dataset import Dataset
from time import time
from Evaluate import _getHitRatio as getHitRatio,_getNDCG as getNDCG
import argparse
def parse_args():
    parser = argparse.ArgumentParser(description="Run NeuMF.")
    parser.add_argument('--path', nargs='?', default='Data/',
                        help='Input data path.')
    parser.add_argument('--dataset', nargs='?', default='ml100k',
                        help='Choose a dataset.')
    parser.add_argument('--epochs', type=int, default=50,
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
class End2endMNCF:
    def __init__(self,sess,num_user,num_item,num_latent,lr,reg,keep_prob):
        self.sess = sess
        self.num_user = num_user
        self.num_item = num_item
        self.num_latent = num_latent
        self.lr = lr
        self.reg = reg
        self.keep_prob = keep_prob
        self.build_up()
    def build_up(self):
        self.uid = tf.placeholder(shape=(None,),dtype=tf.int32)
        self.iid = tf.placeholder(shape=(None,),dtype=tf.int32)
        self.label = tf.placeholder(shape=(None,),dtype=tf.float32)
        self.ubacket = tf.placeholder(shape=(None,None),dtype=tf.int32)
        self.ibacket = tf.placeholder(shape=(None,None),dtype=tf.int32)
        user_embedding = tf.Variable(tf.truncated_normal(shape=(self.num_user,self.num_latent),stddev=0.1),name="user_embedding")
        user_context = tf.Variable(tf.truncated_normal(shape=(self.num_user,self.num_latent),stddev=0.1),name="user_context")
        item_embedding = tf.Variable(tf.truncated_normal(shape=(self.num_item,self.num_latent),stddev=0.1),name="item_embedding")
        item_context = tf.Variable(tf.truncated_normal(shape=(self.num_item,self.num_latent),stddev=0.1),name="item_context")
        ubacket_static = tf.nn.embedding_lookup(item_embedding,self.ubacket)
        ubacket_dynamic = tf.nn.embedding_lookup(item_context,self.ubacket)
        ibacket_static = tf.nn.embedding_lookup(user_embedding,self.ibacket)
        ibacket_dynamic = tf.nn.embedding_lookup(user_context,self.ibacket)
        user_emb = tf.nn.embedding_lookup(user_embedding,self.uid)
        item_emb = tf.nn.embedding_lookup(item_embedding,self.iid)
        user_bias = tf.Variable(tf.constant(0.0,shape=(self.num_user,1)))
        item_bias = tf.Variable(tf.constant(0.0,shape=(self.num_item,1)))
        gb = tf.Variable(tf.constant(0.0,shape=(1,)))
        ub = tf.nn.embedding_lookup(user_bias,self.uid)
        ib = tf.nn.embedding_lookup(item_bias,self.iid)
        # 1 hop 
        uh1 = self.memory_layer(ubacket_static,ubacket_dynamic,item_emb)
        ih1 = self.memory_layer(ibacket_static,ibacket_dynamic,user_emb)
        uo1 = tf.layers.dense(uh1,self.num_latent,activation="relu") + item_emb
        io1 = tf.layers.dense(ih1,self.num_latent,activation="relu") + user_emb
        # 2 hop
        uh2 = self.memory_layer(ubacket_static,ubacket_dynamic,uo1)
        ih2 = self.memory_layer(ibacket_static,ibacket_dynamic,io1)
        uo2 = tf.layers.dense(uh2,self.num_latent,activation="relu") + uo1
        io2 = tf.layers.dense(ih2,self.num_latent,activation="relu") + io1
        output = tf.concat([uo2,io2],axis=1)
        W1 = tf.Variable(tf.truncated_normal(shape=(2*self.num_latent,self.num_latent),stddev=0.1),name="W1")
        b1 = tf.Variable(tf.constant(0.0,shape=(self.num_latent,)),name="b1")
        W2 = tf.Variable(tf.truncated_normal(shape=(self.num_latent,1),stddev=0.1),name="W2")
        b2= tf.Variable(tf.constant(0.0),name="b1")
        h = tf.nn.relu(tf.matmul(output,W1)+b1)
        f = tf.nn.relu(tf.matmul(h,W2)+b2)
        f = f + ub + ib + gb
        #h = tf.layers.dense(output,self.num_latent,activation=tf.nn.relu,name="full_connected_layer1")
        #f = tf.layers.dense(h,1,activation=tf.nn.relu,name="full_connected_layer2")
        self.pred = tf.reduce_sum(f,axis=-1)
        regloss = tf.nn.l2_loss(W1)+tf.nn.l2_loss(W2)
        self.mse = tf.reduce_mean(tf.square(tf.subtract(self.pred,self.label)))
        #self.loss = tf.losses.log_loss(self.pred,self.label) + self.reg*regloss
        self.loss = self.mse + self.reg*regloss
        optimizer = tf.train.AdamOptimizer(learning_rate=self.lr)
        self.train_step = optimizer.minimize(self.loss)
    def memory_layer(self,key,memory,query):
        input = tf.expand_dims(query,axis=1)
        alpha = tf.nn.softmax(tf.reduce_sum(tf.multiply(key,input),axis=-1))
        #alpha = tf.expand_dim(alpha,axis=-1)
        weight = tf.expand_dims(alpha,axis=-1)
        output = tf.reduce_sum(tf.multiply(weight,memory),axis=1)
        return output
    def train(self,batch_users,batch_items,batch_uvecs,batch_ivecs,batch_labels):
        feed_dict = {self.uid:batch_users,self.iid:batch_items,self.ubacket:batch_uvecs,self.ibacket:batch_ivecs,self.label:batch_labels}
        _,loss,y_ = self.sess.run([self.train_step,self.loss,self.pred],feed_dict=feed_dict)
        return loss
    def predict(self,batch_users, batch_items, batch_uvecs, batch_ivecs):
        feed_dict = {self.uid:batch_users,self.iid:batch_items,self.ubacket:batch_uvecs,self.ibacket:batch_ivecs}
        y_ = self.sess.run(self.pred,feed_dict=feed_dict)
        return y_
def generate_train_batch(train,train_matrix,user_len,item_len,item_count,batch_size=512,negative=2):
    batch_users,batch_items,batch_uvecs,batch_ivecs,batch_labels = [],[],[],[],[]
    count= 0
    for u in range(len(train)):
        for i in train[u]:
            uvec = list(np.nonzero(train_matrix[u])[0]+1)
            padd_len = user_len - len(uvec)
            padd_uvec = uvec + [0]*padd_len
            batch_users.append(u)
            batch_uvecs.append(padd_uvec)
            batch_items.append(i)
            col = np.nonzero(train_matrix.T[i])[0]
            ivec = list(col + 1)
            padd_len = item_len - len(ivec)
            padd_ivec = ivec + [0]*padd_len
            batch_ivecs.append(padd_ivec)
            batch_labels.append(1)
            count += 1
            num = 0
            row = np.nonzero(train_matrix[u])[0]
            while num < negative:
                j = np.random.choice(item_count)
                if j not in row:
                    batch_users.append(u)
                    batch_uvecs.append(padd_uvec)
                    batch_items.append(j)
                    col = np.nonzero(train_matrix.T[j])[0]
                    ivec = list(col + 1)
                    padd_len = item_len - len(ivec)
                    padd_ivec = ivec + [0]*padd_len
                    batch_ivecs.append(padd_ivec)
                    batch_labels.append(0)
                    num += 1
                    count += 1
            if count >= batch_size:
                yield batch_users,batch_items,batch_uvecs,batch_ivecs,batch_labels
                batch_users,batch_items,batch_uvecs,batch_ivecs,batch_labels = [], [], [], [], []
                count = 0
    if count >= 0:
        yield batch_users, batch_items, batch_uvecs, batch_masks, batch_labels

def generate_test_batch(test,negatives,train_matrix,user_len,item_len):
    idx = 0
    for u,i in test:
        batch_users, batch_items, batch_uvecs, batch_ivecs = [], [], [], []
        row = np.nonzero(train_matrix[u])[0]
        uvec = list(row+1)
        padd_len = user_len - len(uvec)
        padd_uvec = uvec + [0]*padd_len
        batch_users.append(u)
        batch_uvecs.append(padd_uvec)
        col = np.nonzero(train_matrix.T[i])[0]
        ivec = list(col + 1)
        padd_len = item_len - len(ivec)
        padd_ivec = ivec + [0]*padd_len
        batch_ivecs.append(padd_ivec)
        batch_items.append(i)
        for j in negatives[idx]:
            batch_users.append(u)
            batch_uvecs.append(padd_uvec)
            col = np.nonzero(train_matrix.T[j])[0]
            ivec = list(col + 1)
            padd_len = item_len - len(ivec)
            padd_ivec = ivec + [0]*padd_len
            batch_ivecs.append(padd_ivec)
            batch_items.append(j)
        yield batch_users, batch_items, batch_uvecs, batch_ivecs
        idx += 1
if __name__ == '__main__':
    ds = Dataset('./Data/ml-1m')
    train_matrix = ds.trainMatrix.toarray()
    user_len = np.max(np.sum(train_matrix>0,axis=1))
    item_len = np.max(np.sum(train_matrix.T>0,axis=1))
    print(user_len,item_len)
    num_users = ds.num_users
    num_items = ds.num_items
    args = parse_args()
    topK = 10
    sess = tf.Session()
    model = End2endMNCF(sess,num_users+1,num_items+1,args.num_factors,args.lr,args.reg_lambda,args.keep_prob)
    init = tf.global_variables_initializer()
    sess.run(init)
    init_hits,init_ndcgs = [],[]
    start = time()
    for batch_users, batch_items, batch_uvecs, batch_ivecs in generate_test_batch(ds.testRatings,ds.testNegatives,train_matrix,user_len,item_len):
        scores = model.predict(batch_users, batch_items, batch_uvecs, batch_ivecs)
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
        for batch_users, batch_items, batch_uvecs, batch_ivecs, batch_labels in generate_train_batch(ds.trainList,train_matrix,user_len,item_len,num_items):
            loss = model.train(batch_users,batch_items,batch_uvecs, batch_ivecs, batch_labels)
        # test
        hits = []
        ndcgs = []
        for batch_users, batch_items, batch_uvecs, batch_ivecs in generate_test_batch(ds.testRatings,ds.testNegatives,train_matrix,user_len,item_len):
            scores = model.predict(batch_users,batch_items, batch_uvecs, batch_ivecs)
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




