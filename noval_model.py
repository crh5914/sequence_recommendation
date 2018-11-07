import tensorflow as tf
import numpy as np
from leave_one_dataset import LeaveOneDataset
from time import time
from evaluate import evaluate_model
import argparse
def parse_args():
    parser = argparse.ArgumentParser(description="Run NeuMF.")
    parser.add_argument('--path', nargs='?', default='Data/',
                        help='Input data path.')
    parser.add_argument('--dataset', nargs='?', default='ml100k',
                        help='Choose a dataset.')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of epochs.')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='Batch size.')
    parser.add_argument('--num_factors', type=int, default=64,
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
    def __init__(self,sess,num_users,num_items,num_factors,train_matrix,lr,reg_lambda,keep_prob):
        self.sess = sess
        self.num_users = num_users
        self.num_items = num_items
        self.train_matrix = train_matrix
        self.num_factors = num_factors
        self.reg_lambda = reg_lambda
        self.lr = lr
        self.keep_prob = keep_prob
        self.build_model()
        # self.epochs = epochs
    def build_model(self):
        self.user = tf.placeholder(shape=[None],dtype=tf.int32)
        self.item = tf.placeholder(shape=[None],dtype=tf.int32)
        self.y = tf.placeholder(shape=[None],dtype=tf.float32)
        self.dropout = tf.placeholder(dtype=tf.float32)
        self.rating_matrix = tf.Variable(tf.constant(self.train_matrix),trainable=False)
        self.item_embedding = tf.Variable(tf.random_uniform(shape=(self.num_items,self.num_factors),minval=-0.1,maxval=0.1))
        self.user_embedding = tf.Variable(tf.random_uniform(shape=(self.num_users,self.num_factors),minval=-0.1,maxval=0.1))
        self.fa_W = tf.Variable(tf.random_normal(shape=(3*self.num_factors,self.num_factors)))
        self.fa_b = tf.Variable(tf.constant(0.1,shape=(self.num_factors,)))
        self.it_W = tf.Variable(tf.random_normal(shape=(3*self.num_factors,1)))
        self.it_b = tf.Variable(tf.constant(0.1))
        self.W1 = tf.Variable(tf.random_normal(shape=(3*self.num_factors,int(1.5*self.num_factors))))
        self.b1 = tf.Variable(tf.constant(0.1,shape=(int(1.5*self.num_factors),)))
        self.W2 = tf.Variable(tf.random_normal(shape=(int(1.5*self.num_factors),1)))
        self.b2 = tf.Variable(tf.constant(0.1))
        self.backets = tf.nn.embedding_lookup(self.train_matrix,self.user)
        self.backets = tf.expand_dims(self.backets,-1)
        self.backets_embedding = tf.multiply(self.backets,self.item_embedding)
        self.item_vec = tf.nn.embedding_lookup(self.item_embedding,self.item)
        self.user_vec = tf.nn.embedding_lookup(self.user_embedding,self.user)
        self.factor_attented_backets_vec = self.factor_attention(self.user_vec,self.item_vec,self.backets_embedding,self.fa_W,self.fa_b)
        self.full_attented_backet_vec = self.backets_attention(self.user_vec,self.item_vec,self.factor_attented_backets_vec,self.it_W,self.it_b)
        self.final_vec = tf.concat([self.item_vec,self.user_vec,self.full_attented_backet_vec],axis=1)
        self.f1 = tf.nn.relu(tf.add(tf.matmul(self.final_vec,self.W1),self.b1))
        self.f1 = tf.nn.dropout(self.f1,self.dropout)
        self.y_ = tf.nn.sigmoid(tf.reduce_sum(tf.add(tf.matmul(self.f1,self.W2),self.b2),axis=1))
        self.log_loss = -tf.reduce_mean(self.y * tf.log(tf.clip_by_value(self.y_, 1e-10, 1.0)) + (1 - self.y) * tf.log(tf.clip_by_value(1 - self.y_, 1e-10, 1.0)))
        self.l2_loss = tf.nn.l2_loss(self.W1)
        self.l2_loss += tf.nn.l2_loss(self.b1)
        self.l2_loss += tf.nn.l2_loss(self.W2)
        self.l2_loss += tf.nn.l2_loss(self.b2)
        self.loss = self.log_loss + self.reg_lambda * self.l2_loss
        self.train_opt = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.loss)
        init = tf.global_variables_initializer()
        self.sess.run(init)
    def factor_attention(self,user_vec,item_vec,backets_vec,W,b):
        attented_backets_vec =[]
        for i in range(self.num_items):
            fusion_vec = tf.concat([user_vec,item_vec,backets_vec[:,i,:]],axis=-1)
            alphas = tf.nn.softmax(tf.add(tf.matmul(fusion_vec,W),b))
            new_vec = tf.multiply(alphas,backets_vec[:,i,:])
            attented_backets_vec.append(new_vec)
        attented_vec = tf.concat(attented_backets_vec,axis=1)
        attented_vec = tf.reshape(attented_vec,shape=(-1,self.num_items,self.num_factors))
        return attented_vec
    def backets_attention(self,user_vec,item_vec,backets_vec,W,b):
        betas = []
        for i in range(self.num_items):
            fusion_vec = tf.concat([user_vec, item_vec, backets_vec[:, i, :]], axis=-1)
            beta = tf.nn.softmax(tf.add(tf.matmul(fusion_vec, W), b))
            betas.append(beta)
        betas = tf.concat(betas,axis=1)
        item_attention_weights = tf.nn.softmax(betas)
        item_attention_weights = tf.expand_dims(item_attention_weights,axis=-1)
        aggregated_backet_vec = tf.reduce_sum(tf.multiply(item_attention_weights,backets_vec),axis=1)
        return aggregated_backet_vec
    def fit(self,train_data,batch_size=128,verbose=0):
        users,items,labels = train_data
        for batch_u,batch_v,batch_y in self.generate_train_batch(users,items,labels,batch_size):
            feed_dict = {self.user:batch_u,self.item:batch_v,self.y:batch_y,self.dropout:self.keep_prob}
            _,loss,y_ = self.sess.run([self.train_opt,self.loss,self.y_],feed_dict=feed_dict)
        return loss
    def predict(self,test_data,batch_size=128,verbose=0):
        users,items = test_data
        ys = []
        for batch_u,batch_v in self.generate_test_batch(users,items,batch_size):
            feed_dict = {self.user:batch_u,self.item:batch_v,self.dropout:1.0}
            batch_y = self.sess.run(self.y_,feed_dict=feed_dict)
            print(batch_y)
            ys = ys + list(batch_y)
        return ys
    def generate_train_batch(self,users,items,labels,batch_size):
        batch_u,batch_v,batch_y = [],[],[]
        for u,v,y in zip(users,items,labels):
            batch_u.append(u)
            batch_v.append(v)
            batch_y.append(y)
            if len(batch_u) == batch_size:
                yield batch_u,batch_v,batch_y
                batch_u,batch_v,batch_y = [],[],[]
        if len(batch_u) > 0:
            yield batch_u,batch_v,batch_y
    def generate_test_batch(self,users,items,batch_size):
        batch_u,batch_v= [],[]
        for u,v in zip(users,items):
            batch_u.append(u)
            batch_v.append(v)
            if len(batch_u) == batch_size:
                yield batch_u,batch_v
                batch_u,batch_v = [],[]
        if len(batch_u) > 0:
            yield batch_u,batch_v
def get_train_instances(train_pairs):
    user_input, item_input, labels = [],[],[]
    train_pairs = train_pairs.values
    for pair in train_pairs:
        # positive instance
        u,i = pair[0],pair[1]
        user_input.append(u)
        item_input.append(i)
        labels.append(1)
        # negative instances
        for j in pair[2:]:
            user_input.append(u)
            item_input.append(j)
            labels.append(0)
    return user_input, item_input, labels
if __name__ == '__main__':
    args = parse_args()
    num_epochs = args.epochs
    batch_size = args.batch_size
    num_factors = args.num_factors
    num_negatives = args.num_neg
    reg_lambda = args.reg_lambda
    lr = args.lr
    keep_prob = args.keep_prob
    learner = args.learner
    verbose = args.verbose

    topK = 10
    evaluation_threads = 1  # mp.cpu_count()
    print("NeuMF arguments: %s " % (args))
    model_out_file = 'result/%s_att_%d_%d.h5' % (args.dataset, num_factors, time())

    # Loading data
    t1 = time()
    ds = LeaveOneDataset()
    ds.load('./data/%s' % args.dataset)
    # train, testRatings, testNegatives = dataset.trainMatrix, dataset.testRatings, dataset.testNegatives
    testRatings = ds.test_pairs.values[:, :2]
    testNegatives = ds.test_pairs.values[:, 2:]
    num_users, num_items = ds.num_users, ds.num_items
    train_matrix = np.array(ds.train_matrix.toarray()>0,dtype=np.float32)
    print("Load data done [%.1f s]. #user=%d, #item=%d, #train=%d, #test=%d"
          % (time() - t1, num_users, num_items, len(ds.train_pairs), len(testRatings)))

    # Build model
    sess = tf.Session()
    model = TwoLevelAttetionModel(sess,num_users,num_items,num_factors,train_matrix,lr,reg_lambda,keep_prob)
    # Init performance
    (hits, ndcgs) = evaluate_model(model, testRatings, testNegatives, topK, evaluation_threads)
    hr, ndcg = np.array(hits).mean(), np.array(ndcgs).mean()
    print('Init: HR = %.4f, NDCG = %.4f' % (hr, ndcg))
    best_hr, best_ndcg, best_iter = hr, ndcg, -1
    if args.out > 0:
        model.save_weights(model_out_file, overwrite=True)

        # Training model
    for epoch in range(num_epochs):
        t1 = time()
        # Generate training instances
        user_input, item_input, labels = get_train_instances(ds.train_pairs)

        # Training
        loss = model.fit([user_input, item_input,labels],batch_size=batch_size,verbose=0)
        t2 = time()

        # Evaluation
        if epoch % verbose == 0:
            (hits, ndcgs) = evaluate_model(model, testRatings, testNegatives, topK, evaluation_threads)
            hr, ndcg  = np.array(hits).mean(), np.array(ndcgs).mean()
            print('Iteration %d [%.1f s]: HR = %.4f, NDCG = %.4f, loss = %.4f [%.1f s]'
                  % (epoch, t2 - t1, hr, ndcg, loss, time() - t2))
            if hr > best_hr:
                best_hr, best_ndcg, best_iter = hr, ndcg, epoch
                if args.out > 0:
                    model.save_weights(model_out_file, overwrite=True)

    print("End. Best Iteration %d:  HR = %.4f, NDCG = %.4f. " % (best_iter, best_hr, best_ndcg))
    if args.out > 0:
        print("The best Novel model is saved to %s" % (model_out_file))





