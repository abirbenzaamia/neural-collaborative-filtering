from fedmlp.models import MultiLayerPerceptron as MLP
from fedmlp.train import training_process
from fedmlp.eval import evaluate_model
from fedmlp.config import DEVICE
from dataset import Dataset
from client import Client
from typing import List
import torch.nn
import os
import time


def run_server(dataset: Dataset,
               num_clients: int,
               epochs: int,
               path: str,
               save: bool,
               local_epochs: int,
               args) -> torch.nn.Module:
    """
    defines server side ncf model and initiates the training process
    saves the trained model at indicated path
    """
    # define server side model
    server_model = MLP(dataset.num_users, dataset.num_items)
    server_model.to(DEVICE)
    # print('*********************')
    # print(server_model.state_dict())
    # print('********************')
    
    clients = initialize_clients(dataset)

    # if pretrained model already exists, loads its weights
    # if not, initiates the training process
    if os.path.exists(path):
        trained_weights = torch.load(path)
    else:
        t0 = time.time()
        trained_weights = training_process(server_model, clients, num_clients, epochs, local_epochs, dataset, args)
        t = time.time() - t0
        print(f'The model trained in : {t} (s):, : {t/60:.2f} min')
    if save:
        torch.save(trained_weights, path)
    # load server model's weights to generate recommendations
    server_model.load_state_dict(trained_weights)

    # evaluate the model
    test_data, negatives = dataset.load_test_file(), dataset.load_negative_file()
    assert len(test_data) == len(negatives)
    users, items = zip(*test_data)
    hr, ndcg, mapk, mark = evaluate_model(server_model, users, items, negatives)

    print(f'hit rate: {hr:.2f}, normalized discounted cumulative gain: {ndcg:.2f}')
    print(f'mean average precision @ k: {mapk:.2f}, mean average recall @ k: {mark:.2f}')

    return server_model


def initialize_clients(dataset: Dataset) -> List[Client]:
    """
    creates `Client` instance for each `client_id` in dataset
    returns the list of clients
    """
    clients = list()
    client_dataset = dataset.load_client_train_data()

    for client_id in range(dataset.num_users):
        c = Client(client_id)
        c.set_client_data(client_dataset[client_id])
        clients.append(c)
    return clients
