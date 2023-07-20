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

  def __init__(self,out_path, max_list_len=10, epsilon=1.0, delta = 2.3e-12, num_runs=100):
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
    theta = 5  # initial guess
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

  def client_vote(self, items, r):
    if len(items) < r:
          return 0
    #print(items)
    #print(r)
    if r ==1:
      return 1
    
    #pre = items
    pre = items[r:]
    #print(pre)
    #print(self.server_state.trie)
    verif = False
    i = 0
    while not verif and i< len(pre) :
      #print(pre)
      if(pre[i] in self.server_state.trie):
        verif = True
      i +=1
    if not verif:
      return 0
    # for item in pre:
    #   if item and (item in self.server_state.trie):
    return 1
    # if len(items) < r:
    #   return 0
    # if r == 1:
    #   pre = items[0]
    #   #print(pre)
    #   return 1
    # else:
    #   #print(pre)
    # #print(items)
    #   for _ in items:
    #     pre = _
    #     if pre and (pre in self.server_state.trie):
    #       return 1
    # return 0

  def client_updates(self, r):
    # I encourage you to think about how we could rewrite this function to do
    # one client update (i.e. return 1 vote from 1 chosen client).
    # Then you can have an outer for loop that iterates over chosen clients
    # and calls self.client_update() for each chosen and accumulates the votes.

    votes = defaultdict(int)
    voters = []
    #get voters ids i.e. users 
    for items in random.sample(list(self.clients), self.batch_size):
        voters.append(items)
    #print(voters)

    for items in voters:
        vote_result = self.client_vote(items, r)
        #print(vote_result)
        if vote_result > 0:
            #print(items[0:r])
            # print(items)
            # print(vote_result)
            #print(votes)
            print(items)
            votes[items[0:r][0]] += vote_result
    return votes

  def server_update(self, votes):
    # It might make more sense to define a small class called server_state
    # server_state can track 2 things: 1) updated trie, and 2) quit_sign
    # server_state can be initialized in the constructor of SimulateTrieHH
    # and server_update would just update server_state
    # (i.e, it would update self.server_state.trie & self.server_state.quit_sign)
      
    self.server_state.quit_sign = True
    for prefix in votes:
      if votes[prefix] >= 2:
        print("--------")
        print(votes)
        self.server_state.trie[prefix] = None
        self.server_state.quit_sign = False

    # self.server_state.quit_sign = True
    # #print('-----------------',self.theta)
    # for prefix in votes:
    #   #print(prefix)
    #   if votes[prefix] == 1: 
    #     self.server_state.trie[prefix] = None
    #     self.server_state.quit_sign = False
    #   if votes[prefix] >= self.theta:
    #   #if votes[prefix] >= 2:
    #     self.server_state.trie[prefix] = None
    #     self.server_state.quit_sign = False

  def start(self, batch_size):
    """Implementation of TrieHH."""
    self.server_state.trie.clear()
    r = 1
    while True:
      votes = self.client_updates(r)
      self.server_update(votes)
      #print(votes)
      r +=1
      #r = 10
      if self.server_state.quit_sign or r > self.MAX_L:
        break
      #print(r)

  def get_heavy_hitters(self):
    heavy_hitters = []
    for run in range(self.num_runs):
      self.start(self.batch_size)
      #print(self.server_state.trie)
      raw_result = self.server_state.trie.keys()
      #print(raw_result)
      results = []
      for word in raw_result:
        # if len(word<10):
        results.append(word)
        # if word[-1:] == '$':
        #   results.append(word.rstrip('$'))
      print(f'Discovered {len(results)} heavy hitters in run #{run+1}')
      print(results)
      heavy_hitters.append(results)
    return heavy_hitters