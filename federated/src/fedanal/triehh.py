#
# Created on Thu Jul 13 2023
#
# Copyright (c) 2023 ESI
# @author: ABIR BENZAAMIA (ia_benzaamia@esi.dz)
#

import collections
import math
import pickle
import random
import os

from collections import defaultdict
import numpy as np
import scipy
import scipy.stats
import torch

class ServerState(object):
  def __init__(self):
    self.quit_sign = False
    self.trie = {}

class SimulateTrieHH(object):
  """Simulation for TrieHH."""

  def __init__(self,out_path, max_list_len=10, epsilon=1.0, delta = 2.3e-12, num_runs=10):
    self.MAX_L = max_list_len
    self.delta = delta
    self.epsilon = epsilon
    self.num_runs = num_runs
    self.clients = []
    self.clients_num = 0
    self.server_state = ServerState()
    self._init_clients(out_path)
    self._set_theta()
    self._set_batch_size()

  def _init_clients(self, out_path):
    """Initialization of the dictionary."""
    # make predictions for all clients
    
    with open(os.path.join(out_path, 'predictions.pkl'), 'rb') as fp:
      self.clients = pickle.load(fp)
    #print(self.clients)
    self.clients_num = len(self.clients)
    print(f'Total number of clients: {self.clients_num}')

  def _set_theta(self):
    theta = 4  # initial guess
    delta_inverse = 1 / self.delta
    while ((theta - 3) / (theta - 2)) * math.factorial(theta) < delta_inverse:
      theta += 1
    while theta < np.e ** (self.epsilon/self.MAX_L) - 1:
      theta += 1
    self.theta = theta
    print(f'Theta used by TrieHH: {self.theta}')

  def _set_batch_size(self):
    # check Corollary 1 in our paper.
    # Done in _set_theta: We need to make sure theta >= np.e ** (self.epsilon/self.MAX_L) - 1
    self.batch_size = int( self.clients_num * (np.e ** (self.epsilon/self.MAX_L) - 1)/(self.theta * np.e ** (self.epsilon/self.MAX_L)))
    print(f'Batch size used by TrieHH: {self.batch_size}')


  def client_initialize(self):
    for items in random.sample(list(self.clients), self.batch_size):
        if items[0] in self.server_state.trie:
          #print("--------------------")
          self.server_state.trie[items[0]] += 1
        else:
          self.server_state.trie[items[0]] = 1
        self.server_state.quit_sign = False
        #print(items)
        
  def client_vote(self, items, r):
    #print(self.server_state.trie)
    pre = items[:r]
    #print(pre)
    #print(self.server_state.trie)
    verif = False
    i = 0
    while not verif and i< len(pre) :
      #print(pre)
      if(pre[i] in self.server_state.trie):
        return 1, pre[i]
      i +=1
    if not verif:
      return 0, -1


  def client_updates(self, r):
    votes = defaultdict(int)
    voters = []
    #get recommended items from randomy selected users
    for items in random.sample(list(self.clients), self.batch_size):
        voters.append(items)
    #print(voters)
    for items in voters:
        #print('**********************')
        #print(items)
        vote_result, item = self.client_vote(items, r)
        #vote_results is 1
        if vote_result > 0:
            votes[item] += vote_result
    return votes


  def server_update(self, votes):      
    self.server_state.quit_sign = True
    for item in votes:
      if self.server_state.trie[item] != votes[item]:
        self.server_state.quit_sign = False

  def start(self, r,batch_size):
    """Implementation of TrieHH."""
    results = defaultdict(int)
    self.server_state.trie.clear()
    # intialize clients
    self.client_initialize()
    #print(self.server_state.trie)
    while True:
      votes_before = self.server_state.trie
      votes = self.client_updates(r)
      for i in votes:
        self.server_state.trie[i] += votes[i]
      self.server_update(votes_before)
      if self.server_state.quit_sign:
        break
    print(self.server_state.trie)
    keys_to_remove = []
    for i in self.server_state.trie:
      if self.server_state.trie[i] < 3:
         keys_to_remove.append(i)
        #self.server_state.trie.pop(i, None)
    for key in keys_to_remove:
      del self.server_state.trie[key]



  def get_heavy_hitters(self,r):
    heavy_hitters = []
    for run in range(self.num_runs):
      self.start(r,self.batch_size)
      #print(self.server_state.trie)
      raw_result = self.server_state.trie.keys()
      #print(raw_result)
      results = []
      for item in raw_result:
        # if len(word<10):
        results.append(item)
        # if word[-1:] == '$':
        #   results.append(word.rstrip('$'))
      print(f'Discovered {len(results)} heavy hitters in run #{run+1}')
      print(results)
      heavy_hitters.append(results)

    #print(heavy_hitters)
    one_list = list(set().union(*heavy_hitters))
    final_list = list(one_list)

    return final_list

