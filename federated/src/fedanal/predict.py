from typing import List
import torch 
from fedmlp.config import TOPK
import os
import pickle5 as pickle
import operator
import numpy as np


def predict_all_clients(server_model:torch.nn.Module, client, num_users, num_items, out_path):
    predictions = [[] for u in range(num_users)]
    for u in range(num_users):
        #print(u)
        recommendations = client[u].generate_recommendation(server_model, num_items=num_items, k = TOPK)
        #print(predictions)
        predictions[u] = list(recommendations)
    
    # save predictions file 
    with open(os.path.join(out_path, 'predictions.pkl'), 'wb') as fp:
            pickle.dump(predictions, fp)
    generate_items_fre(predictions, out_path)
    


def generate_items_fre(recommendations, out_path):
       # maximum word length
  max_word_len = 10

  clients_top_items = []
  top_items_counts = {}
  # get the top word for every client
  for client in recommendations:
    top_word = client[0]
    clients_top_items.append(top_word)
    if top_word not in top_items_counts:
      top_items_counts[top_word] = 1
    else:
      top_items_counts[top_word] += 1

  # compute frequencies of top words
  top_word_frequencies = {}
  sum_num = sum(top_items_counts.values())
  for word in top_items_counts:
    top_word_frequencies[word] = top_items_counts[word] * 1.0 / sum_num

  clients_top_items = np.array(clients_top_items)

  with open(os.path.join(out_path, 'items_frequencies.pkl'), 'wb') as fp:
    pickle.dump(top_word_frequencies, fp)
 

