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
    return predictions
    #generate_items_fre(predictions, out_path)
    

def generate_frequency_in_topk(recomendations, k, out_path):
   
  clients_topk_items = []
  top_items_counts = {}

  for reco_lists in recomendations:
     topk_items = reco_lists[:k]
     for i in topk_items:
        clients_topk_items.append(i)
        if i not in top_items_counts:
           top_items_counts[i] = 1
        else:
           top_items_counts[i] +=1

  #compute frequencies for top items
  top_items_frequencies = {}
  sum_num = sum(top_items_counts.values())
  for i in top_items_counts:
    top_items_frequencies[i] = top_items_counts[i] * 1.0 / sum_num

  clients_topk_items = np.array(clients_topk_items)

  with open(os.path.join(out_path, 'items_frequencies.pkl'), 'wb') as fp:
    pickle.dump(top_items_frequencies, fp)

  return top_items_frequencies

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
 

