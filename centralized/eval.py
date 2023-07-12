# from typing import Tuple, List, Optional
# import numpy as np
# import heapq
# import torch
# import math

# from config import DEVICE, TOPK

# def _precision(predicted, actual):
#     prec = [value for value in predicted if value in actual]
#     prec = float(len(prec)) / float(len(predicted))
#     return prec

# def _apk(rank_list: List, item: int) -> float:
#     """
#     Computes the average precision at k.
#     Parameters
#     ----------
#     actual : list
#         A list of actual items to be predicted
#     predicted : list
#         An ordered list of predicted items
#     k : int, default = 10
#         Number of predictions to consider
#     Returns:
#     -------
#     score : float
#         The average precision at k.
#     """
#     predicted = rank_list
#     actual = [item]
#     if not predicted or not actual:
#         return 0.0
    
#     score = 0.0
#     true_positives = 0.0

#     for i, p in enumerate(predicted):
#         if p in actual and p not in predicted[:i]:
#             max_ix = min(i + 1, len(predicted))
#             score += _precision(predicted[:max_ix], actual)
#             true_positives += 1
    
#     if score == 0.0:
#         return 0.0
#     return score / true_positives
    

# def _ark(rank_list: List, item: int):
#     """
#     Computes the average recall at k.
#     Parameters
#     ----------
#     actual : list
#         A list of actual items to be predicted
#     predicted : list
#         An ordered list of predicted items
#     k : int, default = 10
#         Number of predictions to consider
#     Returns:
#     -------
#     score : float
#         The average recall at k.
#     """
#     score = 0.0
#     num_hits = 0.0
#     predicted = rank_list
#     actual = [item]
#     for i,p in enumerate(predicted):
#         if p in actual and p not in predicted[:i]:
#             num_hits += 1.0
#             score += num_hits / (i+1.0)

#     if not actual:
#         return 0.0

#     return score / len(actual)

# def get_metrics(rank_list: List, item: int) -> Tuple[int, float, float, float]:
#     """
#     Used for calculating hit rate & normalized discounted cumulative gain (ndcg)
#     :param rank_list: Top-k list of recommendations
#     :param item: item we are trying to match with `rank_list`
#     :return: tuple containing 1/0 indicating hit/no hit & ndcg & ap@k & ar@k
#     """
#     if item not in rank_list:
#         return 0, 0, 0, 0
#     return 1, math.log(2) / math.log(rank_list.index(item) + 2), _apk(rank_list, item), _ark(rank_list, item)


# def evaluate_model(model: torch.nn.Module,
#                    users: List[int], items: List[int], negatives: List[List[int]],
#                    k: Optional[int] = TOPK) -> Tuple[float, float]:
#     """
#     calculates hit rate and normalized discounted cumulative gain for each user across each item in `negatives`
#     returns average of top-k list of hit rates and ndcgs
#     """

#     hits, ndcgs, apks , arks = list(), list(), list(), list()
#     for user, item, neg in zip(users, items, negatives):

#         item_input = neg + [item]

#         with torch.no_grad():
#             item_input_gpu = torch.tensor(np.array(item_input), dtype=torch.int, device=DEVICE)
#             user_input = torch.tensor(np.full(len(item_input), user, dtype='int32'), dtype=torch.int, device=DEVICE)
#             pred, _ = model(user_input, item_input_gpu)
#             pred = pred.cpu().numpy().tolist()

#         map_item_score = dict(zip(item_input, pred))
#         rank_list = heapq.nlargest(k, map_item_score, key=map_item_score.get)
#         hr, ndcg, apk, ark = get_metrics(rank_list, item)
#         hits.append(hr)
#         ndcgs.append(ndcg)
#         apks.append(apk)
#         arks.append(ark)


#     return np.array(hits).mean(), np.array(ndcgs).mean(), np.array(apks).mean(), np.array(arks).mean()


import heapq # for retrieval topK
import multiprocessing
import numpy as np
from time import time
import math
#from numba import jit, autojit

# Global variables that are shared across processes
_model = None
_testRatings = None
_testNegatives = None
_K = None

def evaluate_model(model, testRatings, testNegatives, K, num_thread):
    """
    Evaluate the performance (Hit_Ratio, NDCG) of top-K recommendation
    Return: score of each test rating.
    """
    global _model
    global _testRatings
    global _testNegatives
    global _K
    _model = model
    _testRatings = testRatings
    _testNegatives = testNegatives
    _K = K
        
    hits, ndcgs, apks, arks  = [], [], [], []
    if(num_thread > 1): # Multi-thread
        pool = multiprocessing.Pool(processes=num_thread)
        res = pool.map(eval_one_rating, range(len(_testRatings)))
        pool.close()
        pool.join()
        hits = [r[0] for r in res]
        ndcgs = [r[1] for r in res]
        apks = [r[2] for r in res]
        arks = [r[3] for r in res]
        return (hits, ndcgs, apks, arks)
    # Single thread
    for idx in range(len(_testRatings)):
        (hr,ndcg, apk, ark) = eval_one_rating(idx)
        hits.append(hr)
        ndcgs.append(ndcg) 
        apks.append(apk)
        arks.append(ark)

             
    return (hits, ndcgs, apks, arks)

def eval_one_rating(idx):
    rating = _testRatings[idx]
    items = _testNegatives[idx]
    u = rating[0]
    gtItem = rating[1][0] 
    items.append(gtItem)
    # Get prediction scores
    map_item_score = {}
    users = np.full(len(items), u, dtype = 'int32')
    predictions = _model.predict([users, np.array(items)], verbose=0)
    for i in range(len(items)):
        item = items[i]
        map_item_score[item] = predictions[i]
    items.pop()
    
    # Evaluate top rank list
    ranklist = heapq.nlargest(_K, map_item_score, key=map_item_score.get)
    hr = getHitRatio(ranklist, gtItem)
    ndcg = getNDCG(ranklist, gtItem)
    apk = getAP_k(ranklist, gtItem)
    ark = getAR_k(ranklist, gtItem)
    return (hr, ndcg, apk, ark)

def getHitRatio(ranklist, gtItem):
    for item in ranklist:
        if item == gtItem:
            return 1
    return 0

def getNDCG(ranklist, gtItem):
    for i in range(len(ranklist)):
        item = ranklist[i]
        if item == gtItem:
            return math.log(2) / math.log(i+2)
    return 0

def getAP_k(ranklist, gtItem):
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
    predicted = ranklist
    actual = [gtItem]
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
    
def getAR_k(ranklist, gtItem):
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
    predicted = ranklist
    actual = [gtItem]
    for i,p in enumerate(predicted):
        if p in actual and p not in predicted[:i]:
            num_hits += 1.0
            score += num_hits / (i+1.0)

    if not actual:
        return 0.0

    return score / len(actual)



def _precision(predicted, actual):
    prec = [value for value in predicted if value in actual]
    prec = float(len(prec)) / float(len(predicted))
    return prec