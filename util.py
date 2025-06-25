import torch
import random
import numpy as np
from torch import nn
from sklearn import metrics

class Myloss(nn.Module):
    def __init__(self, args):
        super(Myloss, self).__init__()
        self.alpha = args.alpha

    def forward(self, one_index, zero_index, target, input):
        loss = nn.MSELoss(reduction='none')
        loss_sum = loss(input, target)
        return (1-self.alpha)*loss_sum[one_index].sum()+self.alpha*loss_sum[zero_index].sum()

def calculate_gauss_sim(adj, gamma: float = 1, axis: int = 0):
    assert len(adj.shape) == 2, "Please input a correct association matrix!"
    if axis == 0:
        kernel = adj @ adj.T
    else:
        kernel = adj.T @ adj

    # get all norms of vectors
    norm = np.diag(kernel)
    norm_sum = norm + norm[:, None]
    return np.exp(-gamma * (norm_sum - 2 * kernel) / np.mean(norm))

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_metrics(score, label):
    y_pre = score
    y_true = label
    metric = caculate_metrics(y_pre, y_true)
    return metric

def caculate_metrics(pre_score, real_score):
    y_true = real_score###
    y_pre = pre_score
    fpr, tpr, thresholds = metrics.roc_curve(y_true, y_pre, pos_label=1)
    auc = metrics.auc(fpr, tpr)
    precision_u, recall_u, thresholds_u = metrics.precision_recall_curve(y_true, y_pre)
    aupr = metrics.auc(recall_u, precision_u)
    # It is used to balance the extremely unbalanced phenomenon caused by high AUC but threshold=0.5 in the sample.
    # sorted_predict_score = np.array(sorted(list(set(np.array(pre_score).flatten()))))
    # sorted_predict_score_num = len(sorted_predict_score)
    # threshold = sorted_predict_score[np.int32(sorted_predict_score_num * np.arange(1, 1000) / 1000)]
    # threshold = np.mean(threshold)
    # threshold = np.mean(thresholds)
    # th_u = (threshold + 0.5) / 2
    y_score = [0 if j < 0.5 else 1 for j in y_pre]


    acc = metrics.accuracy_score(y_true, y_score)
    f1 = metrics.f1_score(y_true, y_score)
    recall = metrics.recall_score(y_true, y_score)
    precision = metrics.precision_score(y_true, y_score)

    metric_result = [auc, aupr, acc, f1, recall, precision]
    print("One epoch metric： ")
    print_met(metric_result)
    return metric_result
    # return auc, acc###


def print_met(list):
    print('AUC ：%.4f ' % (list[0]),
        'AUPR ：%.4f ' % (list[1]),
        'Accuracy ：%.4f ' % (list[2]),
        'f1_score ：%.4f ' % (list[3]),
        'recall ：%.4f ' % (list[4]),
        'precision ：%.4f \n' % (list[5]))