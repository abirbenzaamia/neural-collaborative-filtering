import scipy.stats
import pickle5 as pickle
import numpy as np
import matplotlib.pyplot as plt
from collections import OrderedDict
import sklearn.metrics

class Plot(object):

  def __init__(self, max_k):
    self.confidence = .95


  def _load_true_frequencies(self):
    """Initialization of the dictionary."""
    with open('pretrained/items_frequencies.pkl', 'rb') as fp:
      self.true_frequencies = pickle.load(fp)

  def get_mean_u_l(self, recall_values):
    data_mean = []
    ub = []
    lb = []
    for K in range(1, self.max_k):
      curr_mean = np.mean(recall_values[K])
      data_mean.append(curr_mean)
      n = len(recall_values[K])
      std_err = scipy.stats.sem(recall_values[K])
      h = std_err * scipy.stats.t.ppf((1 + self.confidence) / 2, n - 1)
      lb.append(curr_mean - h)
      ub.append(curr_mean + h)
    mean_u_l = [data_mean, ub, lb]
    print(mean_u_l)
    return mean_u_l


  def precision(self, result, frequencies):
    precision = 0
    for item in result:
      if item in frequencies:
        precision += 1
    precision /= len(result)
    #print('precision', precision)
    return precision
  
  def recall(self, result, frequencies):
    recall = 0
    for i in frequencies:
        if i in result:
          recall +=1
    recall = recall * 1.0/len(result)
    #print('recall', recall)
    return recall


  # calculate F1-score for one list
  def calculate_f1_score(self, triehh_result, frequencies):
    max_k = len(triehh_result)
    sorted_items = sorted(frequencies.items(), key=lambda x: x[1], reverse=True)
    top_items = [item[0] for item in sorted_items]
    top_items = top_items[:max_k]
    print(top_items)
    #sorted_all = OrderedDict(sorted(frequencies, key=lambda x: x[1], reverse = True))
    #top_items = list(sorted_all.keys())[:max_k]
    print(triehh_result)
    print(top_items)
    precision = self.precision(result=triehh_result, frequencies=top_items)
    recall = self.recall(result=triehh_result, frequencies=top_items)
    f1_score = 2*recall*precision/(recall + precision)

    print('precison', precision)
    print('recall', recall)
    print('f1-score', f1_score)

    return precision, recall, f1_score



  def plot_f1_scores(self, precisons, recalls, f1_scores, epsilon, max_k):
  
    all_f1_triehh = []
    k_values = []

    for K in range(1, max_k+1):
      k_values.append(K)
    #caluclate precision and recall

    _, ax1 = plt.subplots(figsize=(10, 7))
    ax1.set_xlabel('K', fontsize=16)
    ax1.set_ylabel('F1 Score', fontsize=16)


    ax1.plot(k_values, f1_scores, color = 'purple', alpha = 1, label=r'TrieHH, $\varepsilon$ = '+str(epsilon))
    #ax1.fill_between(k_values, all_f1_triehh[2], all_f1_triehh[1], color = 'violet', alpha = 0.3)
    plt.legend(loc=4, fontsize=14)

    plt.title('Top K F1 Score', fontsize=14)
    plt.savefig("f1_single.eps")
    plt.savefig("f1_single.png",  bbox_inches="tight")
    plt.close()