import tensorflow as tf
import numpy as np
from evaluate import getHitRatio,getNDCG
from time import time
class Dataset:
    def __init__(self,file,sep):
        self.file = file
        self.sep = sep
        self.build_up()
    def build_up(self):
        self.user2id = {}
        self.item2id = {}
        self.data = {}
        with open(self.file,'r') as fp:
            for line in fp:
                vals = line.strip().split(self.sep)
                u,it,r,t = int(vals[0]),int(vals[1]),float(vals[2]),int(vals[3])
                self.user2id[u] = self.user2id.get(u,len(self.user2id))
                self.item2id[it] = self.item2id.get(it, len(self.item2id))
                if self.user2id[u] not in self.data:
                    self.data[self.user2id[u]] = []
                self.data[self.user2id[u]].append([self.item2id[it],r,t])
        self.length = 0
        for u in self.data:
            sequence = self.data[u]
            sequence = sorted(sequence,key=lambda x: x[2])
            self.data[u] = [x[0] for x in sequence]
            self.length = max(self.length,len(self.data[u]))
        self.num_users = len(self.user2id)
        self.num_items = len(self.item2id)
    def generate_test_batch(self):
        for u in self.data:
            batch_user,batch_sequence,batch_length,batch_item = [],[],[],[]
            its = self.data[u]
            others = np.random.choice(list(set(range(self.num_items))-set(its)),size=99)
            seq = its[:-1] + [0]*(self.length - len(its[:-1]))
            batch_user.append(u)
            batch_item.append(its[-1])
            batch_sequence.append(seq)
            batch_length.append(len(its)-1)
            for o in others:
                batch_user.append(u)
                batch_item.append(o)
                batch_sequence.append(seq)
                batch_length.append(len(its) - 1)
            yield batch_user, batch_sequence, batch_length, batch_item
    def generate_train_batch(self,batch_size=256):
        us = list(self.data.keys())
        np.random.shuffle(us)
        batch_user, batch_sequence, batch_length, batch_item,batch_rating = [], [], [], [], []
        for u in us:
            its = self.data[u]
            for idx in range(len(its[:-1])):
                seq = its[:(idx+1)] + [0] * (self.length - idx - 1)
                batch_user.append(u)
                batch_item.append(its[idx])
                batch_sequence.append(seq)
                batch_length.append(idx+1)
                batch_rating.append(1)
                its = self.data[u]
                others = np.random.choice(list(set(range(self.num_items)) - set(its[:idx+1])), size=4)
                for o in others:
                    batch_user.append(u)
                    batch_item.append(o)
                    batch_sequence.append(seq)
                    batch_length.append(idx+1)
                    batch_rating.append(0)
                if len(batch_user) > batch_size:
                    yield  batch_user,batch_sequence,batch_length,batch_item,batch_rating
                    batch_user, batch_sequence, batch_length, batch_item, batch_rating = [], [], [], [], []
        if len(batch_user) > batch_size:
            yield batch_user, batch_sequence, batch_length, batch_item, batch_rating
class StaticDynamicRecommender:
    def __init__(self,sess,ds,dim,layer_size,lr,topK,epoch):
        self.sess = sess
        self.ds = ds
        self.dim = dim
        self.lr = lr
        self.topK = topK
        self.layer_size = layer_size
        self.epoch = epoch
        self.build_up()
    def build_up(self):
        self.user = tf.placeholder(shape=(None,),dtype=tf.int32)
        self.item = tf.placeholder(shape=(None,),dtype=tf.int32)
        self.sequence = tf.placeholder(shape=(None,None),dtype=tf.int32)
        self.sequence_length = tf.placeholder(shape=(None,),dtype=tf.int32)
        self.rating = tf.placeholder(shape=(None,),dtype=tf.float32)
        user_embedding_table = tf.Variable(tf.truncated_normal(shape=(self.ds.num_users,self.dim),stddev=0.01))
        item_embedding_table = tf.Variable(tf.truncated_normal(shape=(self.ds.num_items, self.dim), stddev=0.01))
        item_sequence_table = tf.Variable(tf.truncated_normal(shape=(self.ds.num_items,self.dim),stddev=0.01))
        user_emb = tf.nn.embedding_lookup(user_embedding_table,self.user)
        item_emb = tf.nn.embedding_lookup(item_embedding_table, self.item)
        sequence_emb = tf.nn.embedding_lookup(item_sequence_table, self.sequence)
        cell = tf.nn.rnn_cell.LSTMCell(num_units=self.dim,state_is_tuple=True)
        _,states = tf.nn.dynamic_rnn(cell=cell,dtype=tf.float32,sequence_length=self.sequence_length,inputs=sequence_emb)
        feat = tf.concat([user_emb,states.h,item_emb],axis=1)
        x = feat
        for i,size in enumerate(self.layer_size[:-1]):
            x = tf.layers.dense(x,size,activation=tf.nn.relu,name='full_connected_layer%d'%i)
        self.y = tf.reduce_sum(tf.layers.dense(x,1,activation=tf.nn.sigmoid,name='prediction_layer'),axis=1)
        self.loss = tf.losses.log_loss(self.y,self.rating)
        opt = tf.train.AdamOptimizer(learning_rate=self.lr)
        self.train_step = opt.minimize(self.loss)
    def train(self):
        init = tf.global_variables_initializer()
        self.sess.run(init)
        s = time()
        hit, ndcg = self.test()
        print('Initializing,hit@{}:{},ndcg@{}:{},time:{}s'.format(self.topK, hit, self.topK, ndcg,time()-s))
        for epoch in range(self.epoch):
            loss,count = 0,0
            s = time()
            for batch_user,batch_sequence,batch_length,batch_item,batch_rating in self.ds.generate_train_batch():
                feed_dict = {self.user:batch_user,self.item:batch_item,self.sequence:batch_sequence,self.sequence_length:batch_length,self.rating:batch_rating}
                _,batch_loss = self.sess.run([self.train_step,self.loss],feed_dict=feed_dict)
                count += len(batch_user)
                loss += batch_loss
            loss = loss / count
            hit,ndcg = self.test()
            print('epoch:{},training loss:{},hit@{}:{},ndcg@{}:{},time:{}s'.format(epoch+1,loss,self.topK,hit,self.topK,ndcg,time()-s))
    def test(self):
        hit,ndcg,count = 0,0,0
        for batch_user, batch_sequence, batch_length, batch_item in self.ds.generate_test_batch():
            feed_dict = {self.user: batch_user, self.item: batch_item, self.sequence: batch_sequence,self.sequence_length: batch_length}
            y = self.sess.run(self.y, feed_dict=feed_dict)
            rank_list = np.argsort(-y)[:self.topK]
            hit += getHitRatio(rank_list,0)
            ndcg += getNDCG(rank_list,0)
            count += 1
        hit = hit / count
        ndcg = ndcg / count
        return hit,ndcg

def main():
    file = './data/ml100k.ratings'
    sep = '\t'
    ds = Dataset(file,sep)
    dim,topK,epoch = 32,10,50
    layer_size = [64,32]
    lr = 0.001
    sess = tf.Session()
    model = StaticDynamicRecommender(sess,ds,dim,layer_size,lr,topK,epoch)
    model.train()
if __name__ == '__main__':
    main()
