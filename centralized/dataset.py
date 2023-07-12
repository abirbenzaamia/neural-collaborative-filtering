#
# Created on Sun Jun 04 2023
#
# Copyright (c) 2023 ESI
# @author: ABIR BENZAAMIA (ia_benzaamia@esi.dz)
#
import scipy.sparse as sp
import pickle5 as pickle
import numpy as np
import pandas as pd
import os
from config import NEG_DATA

class Dataset(object):

    def __init__(self, path):
        '''
        Constructor
        '''
        self.num_users, self.num_items = self.get_data_shape(os.path.join(path,'ratings.csv'), os.path.join(path,'items.csv'))
        self.trainMatrix = self.load_rating_file_as_matrix(os.path.join(path,'train.pkl'))
        self.testRatings = self.load_ratings_as_list(os.path.join(path,'test.pkl'))
        self.testNegatives = self.load_negative_items(os.path.join(path,'negative_items.pkl'))
        assert len(self.testRatings) == len(self.testNegatives)
        
        self.num_users, self.num_items = self.trainMatrix.shape
    
    # read test ratings => one item (leave-one-out)
    def load_ratings_as_list(self, filename):
        ratingList = []
        with open(filename, 'rb') as f:
            ratings_dict = pickle.load(f)
        for user in ratings_dict:
            item = ratings_dict[user]
            ratingList.append([user, item])
        return ratingList
    
    def load_negative_items(self, filename):
        negativeList = []
        with open(filename, 'rb') as f:
            negative_dict = pickle.load(f)
        for user in negative_dict:
            negatives = negative_dict[user][:NEG_DATA]
            negativeList.append(negatives)
        return negativeList
        

    #get train data shape 
    def get_data_shape(self, ratings, items):
        ratings = pd.read_csv(ratings)
        items = pd.read_csv(items)
        num_users = np.unique(ratings['userId'])
        num_items = np.unique(items['itemId'])
        return len(num_users), len(num_items)

    # read train.pkl 
    def load_rating_file_as_matrix(self, filename):
        '''
        Read train.pkl file and Return dok matrix.
        '''
        with open(filename, 'rb') as f:
            train = pickle.load(f)
        self.train = train
        
        mat = sp.dok_matrix((self.num_users, self.num_items), dtype=np.float32)
        for user in train.keys():
            for row in train[user]:
                item, rating = row
                if (rating > 0):
                    mat[user, item] = 1.0
            
        return mat
    
    def get_train(self):
        return self.train



