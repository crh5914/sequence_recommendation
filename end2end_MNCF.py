import tensorflow as tf
from time import time
import argparse
import numpy as np
import logging
def parse_args():
    parser = argparse.ArgumentParser(description="Run MNCF")
    parser.add_argument("--dataset", type=str, default="./data/ml100k.ratings", help="dataset")
    parser.add_argument("--sep", type=str, default="\t", help="delimiter")
    parser.add_argument("--num_factor",type=int,default=32,help="latent factor")
    parser.add_argument("--ratio",type=float,default=0.2,help="test ratio")
    parser.add_argument("--lr",type=float,default=0.001,help="learning rate")
    parser.add_argument("--epochs",type=int,default=50,help="train epochs")
    return parser.parse_args()
class Dataset:
    def __init__(self,file,sep):
        self.file = file
        self.sep = sep
        self.build_up()
    def build_up(self):
        self.user2id = {}
        self.item2id = {}
        self.data = []
        with open(self.file,'r') as fp:
            for line in fp:
                vals = line.strip().split(self.sep)
                u,it,r,t = int(vals[0]),int(vals[1]),float(vals[2]),int(vals[3])
                self.user2id[u] = self.user2id.get(u,len(self.user2id))
                self.item2id[it] = self.item2id.get(it, len(self.item2id))
                self.data.append([self.user2id[u],self.item2id[it],r])
        self.num_user = len(self.user2id)
        self.num_item = len(self.item2id)
    def split(self,test_rate=0.2):
        idx = list(range(len(self.data)))
        np.random.shuffle(idx)
        test_idx = idx[-int(len(self.data)*test_rate):]
        self.train_data,self.test_data = [],[]
        self.udata,self.idata = {},{}
        for i,record in enumerate(self.data):
            if i in test_idx:
                self.test_data.append(record)
            else:
                self.train_data.append(record)
            u,i,r = record
            if u not in self.udata:
                self.udata[u] = []
            self.udata[u].append(i)
            if i not in self.idata:
                self.idata[i] = []
            self.idata[i].append(u)
        self.ulen,self.ilen = 0,0
        for u in self.udata:
            self.ulen = max(self.ulen,len(self.udata[u]))
        for i in self.idata:
            self.ilen = max(self.ilen,len(self.idata[i]))
        del self.data  
    def generate_train_batch(self,batch_size=256):
        np.random.shuffle(self.train_data) 
        ubacket,ibacket,users,items,ratings =[],[],[],[],[]
        for u,i,r in self.train_data:
            users.append(u)
            items.append(i)
            ratings.append(r)
            ubacket.append(self.get_ubacket(u))
            ibacket.append(self.get_ibacket(i))
            if len(users) == batch_size:
                yield ubacket,ibacket,users,items,ratings 
                ubacket,ibacket,users,items,ratings =[],[],[],[],[]
        if len(users) > 0:
            yield ubacket,ibacket,users,items,ratings 
    def get_ibacket(self,i):
        ibk = self.idata[i]
        if len(ibk) < self.ilen:
            ibk = ibk + [self.num_user]*(self.ilen - len(ibk))
        else:
            ibk = ibk[:self.ilen]
        return ibk  
    def get_ubacket(self,u):
        ubk = self.udata[u]
        if len(ubk) < self.ulen:
            ubk = ubk + [self.num_item]*(self.ulen - len(ubk))
        else:
            ubk = ubk[:self.ulen]
        return ubk    
    def generate_test_batch(self,batch_size=256):
        ubacket,ibacket,users,items,ratings =[],[],[],[],[]
        for u,i,r in self.test_data:
            users.append(u)
            items.append(i)
            ratings.append(r)
            ubacket.append(self.get_ubacket(u))
            ibacket.append(self.get_ibacket(i))
            if len(users) == batch_size:
                yield ubacket,ibacket,users,items,ratings 
                ubacket,ibacket,users,items,ratings =[],[],[],[],[]
        if len(users) > 0:
            yield ubacket,ibacket,users,items,ratings
class MNCF:
    def __init__(self,num_user,num_item,num_latent,lr):
        self.num_user = num_user
        self.num_item = num_item
        self.num_latent = num_latent
        self.lr = lr
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
        # 1 hop 
        uh1 = self.memory_layer(ubacket_static,ubacket_dynamic,item_emb)
        ih1 = self.memory_layer(ibacket_static,ibacket_dynamic,user_emb)
        # 2 hop
        uh2 = self.memory_layer(ubacket_static,ubacket_dynamic,uh1)
        ih2 = self.memory_layer(ibacket_static,ibacket_dynamic,ih1)
        output = tf.concat([uh2,ih2],axis=1)
        h = tf.layers.dense(output,self.num_latent,activation=tf.nn.relu,name="full_connected_layer1")
        f = tf.layers.dense(h,1,activation=tf.nn.relu,name="full_connected_layer2")
        self.pred = tf.reduce_sum(f,axis=-1)
        self.loss = tf.reduce_mean(tf.square(tf.subtract(self.pred,self.label)))
        optimizer = tf.train.AdamOptimizer(learning_rate=self.lr)
        self.train_step = optimizer.minimize(self.loss)

    def memory_layer(self,key,memory,query):
        input = tf.expand_dims(query,axis=1)
        alpha = tf.nn.softmax(tf.reduce_sum(tf.multiply(key,input),axis=-1))
        #alpha = tf.expand_dim(alpha,axis=-1)
        weight = tf.expand_dims(alpha,axis=-1)
        output = tf.reduce_sum(tf.multiply(weight,memory),axis=1)
        return output
def get_logger():
    url = 'mncf.log'
    logger = logging.getLogger('MNCF')
    formatter = '%(asctime)s %(levelname)-8s: %(message)s'
    formatter = logging.Formatter(formatter)
    file_handler = logging.FileHandler(url)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.setLevel(logging.INFO)
    return logger
def main():
    args = parse_args()
    ds = Dataset(args.dataset,args.sep)
    ds.split(args.ratio)
    model = MNCF(ds.num_user+1,ds.num_item+1,args.num_factor,args.lr)
    logger = get_logger()
    logger.info('MNCF:(dataset:{},ratio:{},num_user:{},num_item:{},ulen:{},ilen:{},num_factor:{},lr:{})'.format(args.dataset,args.ratio,ds.num_user,ds.num_item,ds.ulen,ds.ilen,args.num_factor,args.lr))
    print('MNCF:(dataset:{},ratio:{},num_user:{},num_item:{},ulen:{},ilen:{},num_factor:{},lr:{})'.format(args.dataset,args.ratio,ds.num_user,ds.num_item,ds.ulen,ds.ilen,args.num_factor,args.lr))
    sess = tf.Session()
    init = tf.global_variables_initializer()
    sess.run(init)
    for epoch in range(args.epochs):
        loss = 0
        for ubacket,ibacket,users,items,ratings in ds.generate_train_batch():
            feed_dict = {model.uid:users,model.iid:items,model.label:ratings,model.ubacket:ubacket,model.ibacket:ibacket}
            _,bloss = sess.run([model.train_step,model.loss],feed_dict=feed_dict)
            loss += bloss*len(ratings)
        train_rmse = np.sqrt(loss/len(ds.train_data))
        mse,mae = 0,0
        for ubacket,ibacket,users,items,ratings in ds.generate_test_batch():
            feed_dict = {model.uid:users,model.iid:items,model.ubacket:ubacket,model.ibacket:ibacket}
            pred = sess.run([model.pred],feed_dict=feed_dict)
            err = np.array(pred) - np.array(ratings)
            mse += np.sum(np.square(err))
            mae += np.sum(np.abs(err))
        rmse,mae = np.sqrt(mse/len(ds.test_data)),mae/len(ds.test_data) 
        print('epoch:{},train rmse:{},test rmse:{},test mae:{}'.format(epoch+1,train_rmse,rmse,mae))
        logger.info('epoch:{},train rmse:{},test rmse:{},test mae:{}'.format(epoch+1,train_rmse,rmse,mae))
if __name__ == '__main__':
    main()
