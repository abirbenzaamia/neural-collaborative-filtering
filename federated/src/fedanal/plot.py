import scipy.stats
import pickle5 as pickle
import numpy as np
import matplotlib.pyplot as plt
from collections import OrderedDict

class Plot(object):

  def __init__(self, max_k):
    self.confidence = .95
    self.max_k = max_k
    self._load_true_frequencies()

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

  def precision(self, result):
    all_words_key = self.true_frequencies.keys()
    precision = 0
    for word in result:
      if word in all_words_key:
        precision += 1
    precision /= len(result)
    print('precision', precision)
    return precision

  def plot_f1_scores(self, triehh_all_results, epsilon):
    # CHANGE "apple" TO "sfp"
    # CLEAN THIS (REMOVE ANY EXCESS CODE NOT USED ANYMORE

    sorted_all = OrderedDict(sorted(self.true_frequencies.items(), key=lambda x: x[1], reverse = True))
    top_words = list(sorted_all.keys())[:self.max_k]

    all_f1_triehh = []
    k_values = []

    for K in range(1, self.max_k):
      k_values.append(K)

    f1_values_triehh = {}
    f1_values_inter = {}

    for K in range(1, self.max_k):
      f1_values_triehh[K] = []
      f1_values_inter[K] = []

    for triehh_result in triehh_all_results:
      for K in range(1, self.max_k):
        recall = 0
        for i in range(K):
          if top_words[i] in triehh_result:
            recall += 1
        recall = recall * 1.0/K
        f1_values_triehh[K].append(2*recall/(recall + 1))
        #print(f1_values_triehh[K])
    all_f1_triehh = self.get_mean_u_l(f1_values_triehh)
    #print(all_f1_triehh)

    _, ax1 = plt.subplots(figsize=(10, 7))
    ax1.set_xlabel('K', fontsize=16)
    ax1.set_ylabel('F1 Score', fontsize=16)


    ax1.plot(k_values, all_f1_triehh[0], color = 'purple', alpha = 1, label=r'TrieHH, $\varepsilon$ = '+str(epsilon))
    #ax1.fill_between(k_values, all_f1_triehh[2], all_f1_triehh[1], color = 'violet', alpha = 0.3)


    plt.legend(loc=4, fontsize=14)

    plt.title('Top K F1 Score ', fontsize=14)
    plt.savefig("f1_single.eps")
    plt.savefig("f1_single.png",  bbox_inches="tight")
    plt.close()