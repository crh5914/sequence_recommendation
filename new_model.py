import tensorflow as tf
import argparse
import numpy as np
import pandas as pd

def parse_args():
    parser = argparse.ArgumentParser(description='run single embedding model')
    parser.add_argument('--num_factors',type=int,default=64,help='item embedding dimension')
    parser.add_argument('--lr',type=float,default=0.001,help='learning rate')
    parser.add_argument('--epochs',type=int,default=50,help='trainning epochs')
    parser.add_argument('--dataset',type=str,default='ml100k',help='specify the trainning dataset')
    paser.add_argument('--sep',type=str,default='\t',help='specify the dilimiter')
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
         self.mask = tf.placeholder(shape=[None,self.user_len],dtype=tf.float32)
         self.target = tf.placeholder(shape=[None,1],dtype=tf.int32)
         self.label = tf.placeholder(shape=[None,1],dtype=tf.float32)
         self.keep_prob = tf.placeholder(dtype=tf.float32)
         with tf.name_scope('embedding'):
             self.Embedding = tf.Variable(tf.random_uniform(shape=[self.num_item,self.num_factor],minval=-0.1,maxval=0.1))
         self.backet_embedding = tf.nn.embedding_lookup(self.Embedding,self.backet)
         self.target_embedding = tf.nn.embedding_lookup(self.Embedding,self.target)
         self.expand_backet_embedding = tf.expand_dims(self.backet_embedding,-1)
         self.pooled_outputs = []
         with tf.name_scope('hconv'):
             for size in hfilter_size:
                 filter_shape = [size,self.num_factor,1,self.hfilter_num]
                 kernal = tf.Variable(tf.random_normal(shape=filter_shape,stddev=0.1))
                 b = tf.Variable(tf.constant(0.1,shape=[self.hfilter_num]))
                 conv = tf.nn.conv2d(self.expand_backet_embedding,kernal,strides=[1,1,1,1],padding='VALID')
                 conv = tf.nn.relu(tf.nn.bias_add(conv,b))
                 pooled = tf.nn.max_pool(conv,ksize=[1,self.user_len-size+1,1,1],strides=[1,1,1,1])
                 self.pooled_outputs.append(pooled)
        num_feat_total = hfilter_num*len(hfilter_size)
        pooled_total = tf.concat(self.pooled_outputs,3)
        self.pooled_flat = tf.reshape(pooled_total,[-1,num_feat_total])
        self.fusion_feat = tf.concat([self.pooled_flat,self.target_embedding],1)
        self.fusion_feat_dropout = tf.nn.dropout(self.fusion_feat,self.keep_prob)
        fusion_dim = num_feat_total + self.num_factors
        self.W1 = tf.Variable(tf.random_normal(shape=[fusion_dim,fusion_dim//2],stddev=0.1))
        self.b1 = tf.Variable(tf.constant(0.1,shape=[fusion_dim//2]))
        self.f1 = tf.nn.relu(tf.add(tf.matmul(self.fusion_feat_dropout,self.W1),self.b1))
        self.W2 = tf.Variable(tf.random_normal(shape=[fusion_dim//2,1],stddev=0.1))
        self.b2 = tf.Variable(tf.constant(0.1))
        self.f2 = tf.nn.sigmoid(tf.add(tf.matmul(self.f1,self.W2),self.b2))
        self.log_loss = -tf.reduce_mean(tf.multiply(self.label,tf.log(self.f2)))
        self.l2_loss = tf.nn.l2_loss(self.W1)
        self.l2_loss += tf.nn.l2_loss(self.b1)
        self.l2_loss += tf.nn.l2_loss(self.W2)
        self.l2_loss += tf.nn.l2_loss(self.b2)
        self.loss = self.log_loss + self.reg_lambda*self.l2_loss
        self.train_opt = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.loss)
        init = tf.global_variables_initializer()
        self.sess.run(init)
    def train(self,backets,targets,labels):
        _,loss,y = self.sess.run([self.train_opt,self.loss,self.f2],feed_dict={self.backet:backets,self.target:targets,self.label:labels,self.keep_prob:0.8})
       return loss,y
    def predict(self,backets,targets):
       y = self.sess.run(self.f2,feed_dict={self.backet:backets,self.target:targets})
       return y
def generate_train_batch(train_matrix,train_pairs,batch_size=256):
    
if __name__ == '__main__':
    pass
        
         
