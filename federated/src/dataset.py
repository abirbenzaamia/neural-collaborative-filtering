from typing import List
import scipy as sp
import numpy as np
import pandas as pd
import pickle5 as pickle
import sys
import os

from fedmlp.config import NUM_NEGATIVES, NEG_DATA

# TODO: convert to pandas


class Dataset:

    def __init__(self, path):
        '''
        Constructor
        '''
        self.path = path
        self.num_users, self.num_items = self.get_data_shape(os.path.join(path,'ratings.csv'), os.path.join(path,'items.csv'))
        self.train_path = os.path.join(path,'train.pkl')
        self.test_path = os.path.join(path,'test.pkl')
        self.neg_path = os.path.join(path,'negative_items.pkl')
        #self.testNegatives = self.load_negative_items(os.path.join(path,'negative_items.pkl'))


    def get_data_shape(self, ratings, items):
        ratings = pd.read_csv(ratings)
        items = pd.read_csv(items)
        num_users = np.unique(ratings['userId'])
        num_items = np.unique(items['itemId'])
        return len(num_users), len(num_items)

    # def load_negative_items(self, filename):
    #         negativeList = [[]]
    #         with open(filename, 'rb') as f:
    #             negative_dict = pickle.load(f)
    #         for user in negative_dict:
    #             negatives = negative_dict[user]
    #             negativeList.append(negatives)
    #         return negativeList

    def load_client_train_data(self) -> List:
        '''
        Loas train data from train.pkl and return user-item matrix
        '''
        with open(self.train_path, 'rb') as f:
            train = pickle.load(f)
        self.train = train

        with open(self.neg_path, 'rb') as f:
            negative = pickle.load(f)
        self.negative = negative

        mat = sp.sparse.dok_matrix((self.num_users, self.num_items), dtype=np.float32)
        for user in train.keys():
            for row in train[user]:
                item, rating = row
                if (rating > 0):
                    mat[user, item] = 1.0

        client_datas = [[[], [], []] for _ in range(self.num_users)]

        for (usr, item) in mat.keys():
            client_datas[usr][0].append(usr)
            client_datas[usr][1].append(item)
            client_datas[usr][2].append(1)
            for t in range(NUM_NEGATIVES):
                neg = np.random.randint(self.num_items)
                while (usr, neg) in mat.keys():
                    neg = np.random.randint(self.num_items)
                client_datas[usr][0].append(usr)
                client_datas[usr][1].append(neg)
                client_datas[usr][2].append(0)
        return client_datas

    def load_test_file(self) -> List[List[int]]:
        ratingList = []
        with open(self.test_path, 'rb') as f:
            ratings_dict = pickle.load(f)
        for user in ratings_dict:
            item = ratings_dict[user][0]
            ratingList.append([user, item])
        return ratingList

    def load_negative_file(self) -> List[List[int]]:
        negativeList = []
        with open(self.neg_path, 'rb') as f:
            negative_dict = pickle.load(f)
        for user in negative_dict:
            negatives = negative_dict[user][:NEG_DATA]
            negativeList.append(negatives)
        return negativeList

    @staticmethod
    def get_movie_names(path, movie_ids: List[int]) -> List[str]:
        movie_names = list()
        items_df = pd.read_csv(os.path.join(path, 'items.csv'))
        items_df.reset_index()
        for index, row in items_df.iterrows():
            name = row["title"] + '--' + row["genres"]
            movie_names.append(name)

        return [movie_names[i] for i in movie_ids]
