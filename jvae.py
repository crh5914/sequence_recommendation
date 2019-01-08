from keras.layers import Input,Dense,Lambda,Concatenate
from keras import Model
from keras import backend as K
from keras import metrics
from leave_one_dataset import LeaveOneDataset
from evaluate import getHitRatio,getNDCG
import argparse
import numpy as np
def parse_args():
    parser = argparse.ArgumentParser(description="Run JVAE")
    parser.add_argument("--dataset", type=str, default="./data/ml100k", help="dataset")
    parser.add_argument("--hidden_dim",type=int,default=100,help="hidden dimension")
    parser.add_argument("--latent_dim",type=int,default=50,help="latent factor dimension")
    parser.add_argument("--epochs",type=int,default=50,help="train epochs")
    return parser.parse_args()

def get_model(num_user,num_item,hidden_dim,latent_dim,stddev):
    x = Input(shape=(num_item,))
    y = Input(shape=(num_user,))
    #r = Input(shape=(1,))
    x_h = Dense(hidden_dim,activation="relu")(x)
    y_h = Dense(hidden_dim,activation="relu")(y)
    x_z_mean = Dense(latent_dim)(x_h)
    x_z_log_var = Dense(latent_dim)(x_h)
    y_z_mean = Dense(latent_dim)(y_h)
    y_z_log_var = Dense(latent_dim)(y_h)
    def sampling(args):
        z_mean, z_log_var = args
        epsilon = K.random_normal(shape=(K.shape(z_mean)[0], latent_dim), mean=0., stddev=stddev)
        return z_mean + K.exp(z_log_var / 2) * epsilon
    x_z = Lambda(sampling,output_shape=(latent_dim,))([x_z_mean,x_z_log_var])
    y_z = Lambda(sampling,output_shape=(latent_dim,))([y_z_mean,y_z_log_var])
    x_d_h = Dense(hidden_dim,activation="relu")(x_z)
    x_d = Dense(num_item,activation="sigmoid")(x_d_h)
    y_d_h = Dense(hidden_dim, activation="relu")(y_z)
    y_d = Dense(num_user, activation="sigmoid")(y_d_h)
    x_mask = K.cast(x>0,"float")
    y_mask = K.cast(y>0,"float")
    x_y = Concatenate(axis=-1)([x_z,y_z])
    f_h = Dense(latent_dim,activation="relu")(x_y)
    f = Dense(1,activation="sigmoid")(f_h)
    xent_loss =  num_item*metrics.binary_crossentropy(x,x_mask*x_d)
    yent_loss =  num_user*metrics.binary_crossentropy(y,y_mask*y_d)
    xkl_loss = - 0.5 * K.sum(1 + x_z_log_var - K.square(x_z_mean) - K.exp(x_z_log_var), axis=-1)
    ykl_loss = - 0.5 * K.sum(1 + y_z_log_var - K.square(y_z_mean) - K.exp(y_z_log_var), axis=-1)
    #rent_loss = metrics.binary_crossentropy(r,f)
    loss = K.mean(xent_loss + xkl_loss + yent_loss + ykl_loss)
    model = Model([x,y],f)
    model.add_loss(loss)
    model.compile(optimizer="adam",loss="binary_crossentropy")
    model.summary()
    return model
def generate_train_batch(train_matrix,train_pairs,batch_size=512):
    batch_users,batch_items,batch_labels = [],[],[]
    count = 0
    for pair in train_pairs:
        u,i = pair[:2]
        uvec = train_matrix[u].copy()
        vvec = train_matrix[:,i].copy()
        uvec[i],vvec[u] = 0,0
        batch_users.append(uvec)
        batch_items.append(vvec)
        batch_labels.append(1)
        count += 1
        for j in pair[2:]:
            uvec = train_matrix[u].copy()
            vvec = train_matrix[:, j].copy()
            uvec[j], vvec[u] = 0, 0
            batch_users.append(uvec)
            batch_items.append(vvec)
            batch_labels.append(0)
            count += 1
        if count >= batch_size:
            yield batch_users,batch_items,batch_labels
            batch_users,batch_items,batch_labels = [],[],[]
            count = 0
    if count >= 0:
        yield batch_users,batch_items,batch_labels
def generate_test_batch(train_matrix,test_pairs):
    for pair in test_pairs:
        u = pair[0]
        batch_users,batch_items = [],[]
        for j in pair[1:]:
            uvec = train_matrix[u].copy()
            vvec = train_matrix[:, j].copy()
            uvec[j], vvec[u] = 0, 0
            batch_users.append(uvec)
            batch_items.append(vvec)
        yield batch_users,batch_items
if __name__ == '__main__':
    args = parse_args()
    ds = LeaveOneDataset()
    ds.load(args.dataset)
    train_matrix = ds.train_matrix.toarray()
    assert np.sum(train_matrix) == np.sum(train_matrix > 0)
    #train_matrix[1]
    topK = 10
    num_user = ds.num_users
    num_item = ds.num_items
    hidden_dim = args.hidden_dim
    latent_dim = args.latent_dim
    model = get_model(num_user,num_item,hidden_dim,latent_dim,1.0)
    best_hit = 0
    best_ndcg = 0
    for epoch in range(args.epochs):
        #train
        for batch_users,batch_items,batch_labels in generate_train_batch(train_matrix,ds.train_pairs.values):
            model.train_on_batch([batch_users,batch_items],batch_labels)
        # test
        hits = []
        ndcgs = []
        for batch_users,batch_items in generate_test_batch(train_matrix,ds.test_pairs.values):
            scores = model.predict_on_batch([batch_users,batch_items])
            print(scores)
            scores = np.reshape(scores,-1)
            ranklist = np.argsort(-scores)[:topK]
            hits.append(getHitRatio(ranklist,0))
            ndcgs.append(getNDCG(ranklist,0))
        hit = np.mean(hits)
        ndcg = np.mean(ndcgs)
        print('test epoch:',epoch,'hit@{}:{},ndcg@{}:{}'.format(topK,hit,topK,ndcg))
        if hit > best_hit:
            best_hit = hit
        if ndcg > best_ndcg:
            best_ndcg = ndcg
    print('best hit@{}:{},best ndcg@{}:{}'.format(topK,best_hit,topK,best_ndcg))



