import tensorflow as tf
import argparse
from leave_one_dataset import LeaveOneDataset
from evaluate import getHitRatio,getNDCG
import numpy as np
import pandas as pd

def parse_args():
    parser = argparse.ArgumentParser(description='run single embedding model')
    parser.add_argument('--num_factors',type=int,default=64,help='item embedding dimension')
    parser.add_argument('--lr',type=float,default=0.001,help='learning rate')
    parser.add_argument('--epochs',type=int,default=50,help='trainning epochs')
    parser.add_argument('--hfilter_num',type=int,default=50,help='horizatal filter number')
    parser.add_argument('--vfilter_num',type=int,default=50,help='vertical filter number')
    parser.add_argument('--hfilter_size',nargs='?',default='[2,3]',help='hfilter_size')
    parser.add_argument('--vfilter_size',nargs='?',default='[2,3]',help='vfilter_size')
    parser.add_argument('--reg_lambda',type=float,default=0.01,help='regurizer rate')
    parser.add_argument('--dataset',type=str,default='ml100k',help='specify the trainning dataset')
    parser.add_argument('--sep',type=str,default='\t',help='specify the dilimiter')
    return parser.parse_args()
class Model:
    def __init__(self,sess,num_item,user_len,num_factor,hfilter_size,hfilter_num,vfilter_size,vfilter_num,lr,epochs,reg_lambda):
         self.sess = sess
         self.num_item = num_item
         self.user_len = user_len
         self.num_factor = num_factor
         self.hfilter_size = hfilter_size
         self.hfilter_num = hfilter_num
         self.vfilter_size = vfilter_size
         self.vfilter_num = vfilter_num
         self.lr = lr
         self.epochs = epochs
         self.reg_lambda = reg_lambda
         self.build_graph()
    def build_graph(self):
         self.backet = tf.placeholder(shape=[None,self.user_len],dtype=tf.int32)
         #self.mask = tf.placeholder(shape=[None,self.user_len],dtype=tf.float32)
         self.target = tf.placeholder(shape=[None,1],dtype=tf.int32)
         self.label = tf.placeholder(shape=[None,1],dtype=tf.float32)
         self.keep_prob = tf.placeholder(dtype=tf.float32)
         self.Embedding = tf.Variable(tf.random_uniform(shape=[self.num_item,self.num_factor],minval=-0.1,maxval=0.1))
         self.backet_embedding = tf.nn.embedding_lookup(self.Embedding,self.backet)
         self.target_embedding = tf.nn.embedding_lookup(self.Embedding,self.target)
         self.expand_backet_embedding = tf.expand_dims(self.backet_embedding,-1)
         self.pooled_outputs = []
         for size in self.hfilter_size:
             filter_shape = [size,self.num_factor,1,self.hfilter_num]
             kernal = tf.Variable(tf.random_normal(shape=filter_shape,stddev=0.1))
             b = tf.Variable(tf.constant(0.1,shape=[self.hfilter_num]))
             conv = tf.nn.conv2d(self.expand_backet_embedding,kernal,strides=[1,1,1,1],padding='VALID')
             conv = tf.nn.relu(tf.nn.bias_add(conv,b))
             pooled = tf.nn.max_pool(conv,ksize=[1,self.user_len-size+1,1,1],strides=[1,1,1,1],padding='VALID')
             self.pooled_outputs.append(pooled)
         num_feat_total = self.hfilter_num*len(self.hfilter_size)
         pooled_total = tf.concat(self.pooled_outputs,3)
         self.pooled_flat = tf.reshape(pooled_total,[-1,num_feat_total])
         self.target_embedding = tf.reshape(self.target_embedding,[-1,self.num_factor])
         self.fusion_feat = tf.concat([self.pooled_flat,self.target_embedding],1)
         self.fusion_feat_dropout = tf.nn.dropout(self.fusion_feat,self.keep_prob)
         fusion_dim = num_feat_total + self.num_factor
         self.W1 = tf.Variable(tf.random_normal(shape=[fusion_dim,fusion_dim//2],stddev=0.1))
         self.b1 = tf.Variable(tf.constant(0.1,shape=[fusion_dim//2]))
         self.f1 = tf.nn.relu(tf.add(tf.matmul(self.fusion_feat_dropout,self.W1),self.b1))
         self.W2 = tf.Variable(tf.random_normal(shape=[fusion_dim//2,1],stddev=0.1))
         self.b2 = tf.Variable(tf.constant(0.1))
         self.f2 = tf.nn.sigmoid(tf.add(tf.matmul(self.f1,self.W2),self.b2))
         self.log_loss = -tf.reduce_mean(self.label* tf.log(tf.clip_by_value(self.f2,1e-10,1.0))+(1-self.label) * tf.log(tf.clip_by_value(1-self.f2,1e-10,1.0)))
         self.l2_loss = tf.nn.l2_loss(self.W1)
         self.l2_loss += tf.nn.l2_loss(self.b1)
         self.l2_loss += tf.nn.l2_loss(self.W2)
         self.l2_loss += tf.nn.l2_loss(self.b2)
         self.loss = self.log_loss + self.reg_lambda*self.l2_loss
         self.train_opt = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.loss)
         init = tf.global_variables_initializer()
         self.sess.run(init)
    def train(self,backets,targets,labels):
        _,loss,y = self.sess.run([self.train_opt,self.loss,self.f2],feed_dict={self.backet:backets,self.target:np.reshape(np.array(targets),(-1,1)),self.label:np.reshape(np.array(labels),(-1,1)),self.keep_prob:0.8})
        return loss,y
    def predict(self,backets,targets):
        y = self.sess.run(self.f2,feed_dict={self.backet:backets,self.target:np.reshape(np.array(targets),(-1,1)),self.keep_prob:1.0})
        return y
def generate_train_batch(train_matrix,train_pairs,user_len,num_items,batch_size=512):
    batch_users,batch_items,batch_labels = [],[],[]
    count = 0
    for pair in train_pairs:
        u,i = pair[:2]
        uvec = list(np.nonzero(train_matrix[u])[0])
        padd_len = user_len - len(uvec)
        padd_uvec = uvec + [num_items]*padd_len
        batch_users.append(padd_uvec)
        batch_items.append(i)
        batch_labels.append(1)
        count += 1
        for j in pair[2:]:
            batch_users.append(padd_uvec)
            batch_items.append(j)
            batch_labels.append(0)
            count += 1
        if count >= batch_size:
            yield batch_users,batch_items,batch_labels
            batch_users,batch_items,batch_labels = [],[],[]
            count = 0
    if count >= 0:
        yield batch_users,batch_items,batch_labels
        batch_users,batch_items,batch_labels = [],[],[]
        count = 0
def generate_test_batch(train_matrix,test_pairs,user_len,num_items):
    for pair in test_pairs:
        u = pair[0]
        batch_users,batch_items = [],[]
        uvec = list(np.nonzero(train_matrix[u])[0])
        padd_len = user_len - len(uvec)
        padd_uvec = uvec + [num_items]*padd_len
        #batch_users.append(padd_uvec)
        for j in pair[1:]:
            batch_users.append(padd_uvec)
            batch_items.append(j)
        yield batch_users,batch_items
if __name__ == '__main__':
    ds = LeaveOneDataset()
    ds.load('./data/ml100k')
    train_matrix = ds.train_matrix.toarray()
    user_len = np.max(np.sum(train_matrix>0,axis=1))
    args = parse_args()
    hfilter_size = eval(args.hfilter_size)
    vfilter_size = eval(args.vfilter_size)
    topK = 10
    sess = tf.Session()
    model = Model(sess,ds.num_items+1,user_len,args.num_factors,hfilter_size,args.hfilter_num,vfilter_size,args.vfilter_num,args.lr,args.epochs,args.reg_lambda)
    best_hit = 0
    best_ndcg = 0
    for epoch in range(args.epochs):
        #train
        for batch_users,batch_items,batch_labels in generate_train_batch(train_matrix,ds.train_pairs.values,user_len,ds.num_items):
            loss,_ = model.train(batch_users,batch_items,batch_labels)
        # test
        hits = []
        ndcgs = []
        for batch_users,batch_items in generate_test_batch(train_matrix,ds.test_pairs.values,user_len,ds.num_items):
            scores = model.predict(batch_users,batch_items)
            scores = np.reshape(scores,-1)
            ranklist = np.argsort(-scores)[:topK]
            hits.append(getHitRatio(ranklist,0))
            ndcgs.append(getNDCG(ranklist,0))
        hit = np.mean(hits)
        ndcg = np.mean(ndcgs)
        print('train epoch:',epoch,'loss:',loss)
        print('test epoch:',epoch,'hit@{}:{},ndcg@{}:{}'.format(topK,hit,topK,ndcg))
        if hit > best_hit:
            best_hit = hit
        if ndcg > best_ndcg:
            best_ndcg = ndcg
    print('best hit@{}:{},best ndcg@{}:{}'.format(topK,best_hit,topK,best_ndcg))
        
    
        
         
