from typing import Tuple, List, Optional
import numpy as np
import heapq
import torch
import math


from fedmlp.config import DEVICE, TOPK

def _precision(predicted, actual):
    prec = [value for value in predicted if value in actual]
    prec = float(len(prec)) / float(len(predicted))
    return prec

def _apk(rank_list: List, item: int) -> float:
    """
    Computes the average precision at k.
    Parameters
    ----------
    actual : list
        A list of actual items to be predicted
    predicted : list
        An ordered list of predicted items
    k : int, default = 10
        Number of predictions to consider
    Returns:
    -------
    score : float
        The average precision at k.
    """
    predicted = rank_list
    actual = [item]
    if not predicted or not actual:
        return 0.0
    
    score = 0.0
    true_positives = 0.0

    for i, p in enumerate(predicted):
        if p in actual and p not in predicted[:i]:
            max_ix = min(i + 1, len(predicted))
            score += _precision(predicted[:max_ix], actual)
            true_positives += 1
    
    if score == 0.0:
        return 0.0
    return score / true_positives
    

def _ark(rank_list: List, item: int):
    """
    Computes the average recall at k.
    Parameters
    ----------
    actual : list
        A list of actual items to be predicted
    predicted : list
        An ordered list of predicted items
    k : int, default = 10
        Number of predictions to consider
    Returns:
    -------
    score : float
        The average recall at k.
    """
    score = 0.0
    num_hits = 0.0
    predicted = rank_list
    actual = [item]
    for i,p in enumerate(predicted):
        if p in actual and p not in predicted[:i]:
            num_hits += 1.0
            score += num_hits / (i+1.0)

    if not actual:
        return 0.0

    return score / len(actual)

def get_metrics(rank_list: List, item: int) -> Tuple[int, float, float, float]:
    """
    Used for calculating hit rate & normalized discounted cumulative gain (ndcg)
    :param rank_list: Top-k list of recommendations
    :param item: item we are trying to match with `rank_list`
    :return: tuple containing 1/0 indicating hit/no hit & ndcg & ap@k & ar@k
    """
    if item not in rank_list:
        return 0, 0, 0, 0
    return 1, math.log(2) / math.log(rank_list.index(item) + 2), _apk(rank_list, item), _ark(rank_list, item)


def evaluate_model(model: torch.nn.Module,
                   users: List[int], items: List[int], negatives: List[List[int]],
                   k: Optional[int] = TOPK) -> Tuple[float, float]:
    """
    calculates hit rate and normalized discounted cumulative gain for each user across each item in `negatives`
    returns average of top-k list of hit rates and ndcgs
    """

    hits, ndcgs, apks , arks = list(), list(), list(), list()
    for user, item, neg in zip(users, items, negatives):

        item_input = neg + [item]

        with torch.no_grad():
            item_input_gpu = torch.tensor(np.array(item_input), dtype=torch.int, device=DEVICE)
            user_input = torch.tensor(np.full(len(item_input), user, dtype='int32'), dtype=torch.int, device=DEVICE)
            pred, _ = model(user_input, item_input_gpu)
            pred = pred.cpu().numpy().tolist()

        map_item_score = dict(zip(item_input, pred))
        rank_list = heapq.nlargest(k, map_item_score, key=map_item_score.get)
        hr, ndcg, apk, ark = get_metrics(rank_list, item)
        hits.append(hr)
        ndcgs.append(ndcg)
        apks.append(apk)
        arks.append(ark)


    return np.array(hits).mean(), np.array(ndcgs).mean(), np.array(apks).mean(), np.array(arks).mean()
