# !/bin/python3
import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
import os
class LeaveOneDataset:
    def __init__(self):
        pass
    def build_up(self,file,sep='\t',fields=['user_id','item_id','rating','timestamp']):
        print('build up dataset from ',file)
        df = pd.read_csv(file,sep=sep)
        df.columns = fields
        df = df.sort_values(['user_id','timestamp']).reset_index(drop=True)
        user_count = df.groupby(['user_id'],as_index=False).count()
        # item_count = df.groupby(['item_id',as_index=False]).count()        
        self.unique_users = pd.unique(df['user_id'])
        self.unique_items = pd.unique(df['item_id'])
        self.user_map = dict({id:i for i,id in enumerate(self.unique_users)})
        self.item_map = dict({id:i for i,id in enumerate(self.unique_items)})
        self.num_users = len(self.unique_users)
        self.num_items = len(self.unique_items)
        print('total users:',self.num_users,' items:',self.num_items)
        last_index = np.cumsum(user_count['item_id']) - 1
        self.test_df = self.map_field(df[df.index.isin(last_index)])
        self.train_df = self.map_field(df[~df.index.isin(last_index)])
        self.train_matrix =  csr_matrix((self.train_df['rating'].values,(self.train_df['user_id'].values,self.train_df['item_id'].values)),shape=(self.num_users,self.num_items))
    def generate_pair(self,train_neg_samples=1,test_neg_samples=99):
        self.train_pairs = []
        self.test_pairs = []
        cols = self.train_matrix.indices
        row = self.train_matrix.indptr
        test_uids = self.test_df['user_id'].values
        for uid in range(self.num_users):
            its = cols[row[uid]:row[uid+1]]
            left_items = list(set(range(self.num_items))-set(its))
            # print(left_items)
            for it in its:
                #train_samples = [uid,it]
                # self.pos_pairs.append([uid,it,1])
                neg_items = []
                count = 0
                while count < train_neg_samples:
                    neg_it = np.random.choice(left_items)
                    if neg_it in neg_items:
                        continue
                    neg_items.append(neg_it)
                    count += 1
                neg_items = [uid,it] + neg_items
                self.train_pairs.append(neg_items)
            if uid in test_uids:
                neg_samples = [uid]
                neg_samples.append(int(self.test_df.loc[self.test_df['user_id']==uid,'item_id']))
                count = 0
                while count < test_neg_samples:
                    neg_it = np.random.choice(left_items)
                    if neg_it in neg_samples:
                        continue
                    neg_samples.append(neg_it)
                    count += 1
                self.test_pairs.append(neg_samples)
    def save(self,dir):
        if not os.path.exists(dir):
            os.makedirs(dir)
        self.test_df.to_csv(os.path.join(dir,'test_data.csv'),index=False)
        self.train_df.to_csv(os.path.join(dir,'train_data.csv'),index=False)
        # pd.DataFrame(self.pos_pairs).to_csv(os.path.join(dir,'train_pos_pairs.csv'),index=False,header=False)
        pd.DataFrame(self.train_pairs).to_csv(os.path.join(dir,'train_pairs.csv'),index=False,header=False)
        pd.DataFrame(self.test_pairs).to_csv(os.path.join(dir,'test_pairs.csv'),index=False,header=False)
        pd.DataFrame({'unique_users':self.unique_users}).to_csv(os.path.join(dir,'unique_users.csv'),index=False,header=False)
        pd.DataFrame({'unique_items':self.unique_items}).to_csv(os.path.join(dir,'unique_items.csv'),index=False,header=False)
        
    def load(self,dir):
        # self.pos_pairs = pd.read_csv(os.path.join(dir,'train_pos_pairs.csv'),header=None)
        self.train_pairs = pd.read_csv(os.path.join(dir,'train_pairs.csv'),header=None)
        self.test_pairs = pd.read_csv(os.path.join(dir,'test_pairs.csv'),header=None)
        # print(self.test_pairs.describe())
        self.num_users = len(pd.read_csv(os.path.join(dir,'unique_users.csv'),header=None))
        self.num_items = len(pd.read_csv(os.path.join(dir,'unique_items.csv'),header=None))
        self.test_df = pd.read_csv(os.path.join(dir,'test_data.csv'))
        self.train_df = pd.read_csv(os.path.join(dir,'train_data.csv'))
        self.train_matrix =  csr_matrix((self.train_df['rating'].values,(self.train_df['user_id'].values,self.train_df['item_id'].values)),shape=(self.num_users,self.num_items))

    def map_field(self,tp):
        # print(tp['item_id'].apply(lambda x: self.item_map[int(x)]))
        tp['item_id'] = tp['item_id'].apply(lambda x: self.item_map[int(x)])
        # print('end')
        tp['user_id'] = tp['user_id'].apply(lambda x: self.user_map[int(x)])
        return tp
if __name__ == '__main__':
    file = './data/ml1m.ratings'
    ds = LeaveOneDataset()
    ds.build_up(file,sep="::")  
    ds.generate_pair(train_neg_samples=4)
    ds.save('./data/ml1m')
    # ds.load('./data/ml100k/')
    print(ds.num_items,ds.num_users)


