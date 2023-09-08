
from fedmf.models import MatrixFactorization as MF
from fedmf.train import training_process
from fedmf.config import DEVICE, TOPK
from dataset import Dataset
from client import Client
from typing import List
import torch.nn
import os
import time
import torch
from torch import nn
from torch import optim 

from typing import List, Optional, Any, Tuple
from sklearn.cluster import KMeans

def kmeans(embeddings, num_clusters, num_iterations):
    # Convert the list of embeddings to a PyTorch tensor
    
    # Perform k-means clustering using sklearn KMeans
    kmeans = KMeans(n_clusters=num_clusters, max_iter=num_iterations, random_state=0, n_init=10)
    kmeans.fit(embeddings)
    
    # Get the cluster labels and cluster centroids
    cluster_labels = kmeans.labels_
    cluster_centroids = torch.tensor(kmeans.cluster_centers_)
    
    return cluster_centroids, cluster_labels

def get_similar_items(server_model: torch.nn.Module, num_users: int,
                                num_items: int, dataset, raw_path, k: Optional[int] = TOPK) -> List[int]:
    movies = set(range(num_items))
    movies = torch.tensor(list(movies), dtype=torch.int, device=DEVICE)
    weights = server_model.state_dict()
    print(weights['mf_embedding_item.weight'])
    print(type(weights['mf_embedding_item.weight']))
    print(weights['mf_embedding_item.weight'].shape)
    print('------------------------')
    embeddings = weights['mf_embedding_item.weight'].clone().detach().cpu()
    #embeddings.requires_grad_(True)
    num_clusters = 100
    num_iterations = 100

    centroids, cluster_labels = kmeans(embeddings, num_clusters, num_iterations)
    print(cluster_labels.shape)

    #get group of items with names
    group = {e:[] for e in range(100)}

    for i in range(100):
        indices = [index for index, element in enumerate(cluster_labels) if element == i]
        group[i].append(indices)

    #print(group)

    for i in range(100):
        print('------------ the {} group '.format(i))
        for l in group[i]:
            print(dataset.get_movie_names(raw_path, l))
        print('---------------------------------------')
    client_id = torch.tensor([1 for _ in range(len(movies))], dtype=torch.int, device=DEVICE)
    #logits, _ = server_model(client_id, movies)
    #print(logits.shape)
    #print(logits)
# def generate_recommendation(self, server_model: torch.nn.Module,
#                                 num_items: int,  k: Optional[int] = TOPK) -> List[int]:
#         """
#         :param server_model: server model which will be used to generate predictions
#         :param num_items: total number of unique items in dataset
#         :param k: number of recommendations to generate
#         :return: list of `k` movie recommendations
#         """
#         # get movies that user has not yet interacted with
#         movies = set(range(num_items)).difference(set(self.client_data['item_id'].tolist()))
#         movies = torch.tensor(list(movies), dtype=torch.int, device=DEVICE)
#         client_id = torch.tensor([self.client_id for _ in range(len(movies))], dtype=torch.int, device=DEVICE)
#         # obtain predictions in terms of logit per movie
#         with torch.no_grad():
#             logits, _ = server_model(client_id, movies)

#         rec_dict = {movie: p for movie, p in zip(movies.tolist(), logits.squeeze().tolist())}
#         # select top k recommendations
#         top_k = sorted(rec_dict.items(), key=lambda x: -x[1])[:k]
#         rec, _ = zip(*top_k)

#         return rec

# def heavy_hitters():

