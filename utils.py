# -*- coding: utf-8 -*-
""" Set of utilities """
from torch import nn
import torch
from torch.autograd import Variable
import numpy as np
from torch.nn.modules.loss import _Loss
from torch.nn import functional as F

class MeanTopKRecallMeter(object):
    def __init__(self, num_classes, k=5):
        self.num_classes = num_classes
        self.k = k
        self.reset()

    def reset(self):
        self.tps = np.zeros(self.num_classes)
        self.nums = np.zeros(self.num_classes)

    def add(self, scores, labels):
        tp = (np.argsort(scores, axis=1)[:, -self.k:] == labels.reshape(-1, 1)).max(1)
        for l in np.unique(labels):
            self.tps[l]+=tp[labels==l].sum()
            self.nums[l]+=(labels==l).sum()

    def value(self):
        recalls = (self.tps/self.nums)[self.nums>0]
        if len(recalls)>0:
            return recalls.mean()*100
        else:
            return None

        
class ValueMeter(object):
    def __init__(self):
        self.sum = 0
        self.total = 0

    def add(self, value, n):
        self.sum += value * n
        self.total += n

    def value(self):
        return self.sum / self.total


def topk_accuracy(scores, labels, ks, selected_class=None):
    """Computes TOP-K accuracies for different values of k
    Args:
        scores: numpy nd array, shape = (instance_count, label_count)
        labels: numpy nd array, shape = (instance_count,)
        ks: tuple of integers
    Returns:
        list of float: TOP-K accuracy for each k in ks
    """
    if selected_class is not None:
        idx = labels == selected_class
        scores = scores[idx]
        labels = labels[idx]
    rankings = scores.argsort()[:, ::-1]
    maxk = np.max(ks)  # trim to max k to avoid extra computation

    # compute true positives in the top-maxk predictions
    tp = rankings[:, :maxk] == labels.reshape(-1, 1)

    # trim to selected ks and compute accuracies
    return [tp[:, :k].max(1).mean() for k in ks]


def topk_accuracy_save_validation_pred(scores, labels, ks, modality, no_classes = 2513, selected_class=None):
    """Computes TOP-K accuracies for different values of k
    Args:
        scores: numpy nd array, shape = (instance_count, label_count)
        labels: numpy nd array, shape = (instance_count,)
        ks: tuple of integers

    Returns:
        list of float: TOP-K accuracy for each k in ks
    """
    if selected_class is not None:
        idx = labels == selected_class
        scores = scores[idx]
        labels = labels[idx]
    ranking = scores.argsort()[:, ::-1]
    maxk = np.max(ks)  # trim to max k to avoid extra computation

    # compute true positives in the top-maxk predictions
    tp = ranking[:, :maxk] == labels.reshape(-1, 1)

    allzs = np.zeros((no_classes,), dtype=int)
    allzs_correct = np.zeros((no_classes,), dtype=int)
    for aa in range(len(labels)):
        curr_label = labels[aa]
        curr_pred = ranking[:, :maxk][aa][0]
        allzs[curr_label] = allzs[curr_label] + 1
        if curr_label == curr_pred:
            allzs_correct[curr_label] = allzs_correct[curr_label] + 1

    for aa in range(no_classes):
        with open('validation_pred_'+str(modality)+'.txt', 'a') as f:
            f.write("%d\t%d\n" % (allzs_correct[aa], allzs[aa]))

    # trim to selected ks and compute accuracies
    return [tp[:, :k].max(1).mean() for k in ks]


def topk_recall(scores, labels, k=5, classes=None):
    unique = np.unique(labels)
    if classes is None:
        classes = unique
    else:
        classes = np.intersect1d(classes, unique)
    recalls = 0

    for c in classes:
        recalls += topk_accuracy(scores, labels, ks=(k,), selected_class=c)[0]
    return recalls / len(classes)


'''def topk_recall_multiple_timesteps(preds, labels, k=5, classes=None):
    accs = np.array([topk_recall(preds[:, t, :], labels, k, classes)
                     for t in range(preds.shape[1])])
    return accs.reshape(1, -1)'''


def get_marginal_indexes(actions, mode):
    """For each verb/noun retrieve the list of actions containing that verb/name
        Input:
            mode: "verb" or "noun"
        Output:
            a list of numpy array of indexes. If verb/noun 3 is contained in actions 2,8,19,
            then output[3] will be np.array([2,8,19])
    """
    vi = []
    for v in range(actions[mode].max() + 1):
        vals = actions[actions[mode] == v].index.values
        if len(vals) > 0:
            vi.append(vals)
        else:
            vi.append(np.array([0]))
    return vi


def marginalize(probs, indexes):
    mprobs = []
    for ilist in indexes:
        mprobs.append(probs[:, ilist].sum(1))
    return np.array(mprobs).T


def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    xx = x
    x = x.reshape((-1, x.shape[-1]))
    e_x = np.exp(x - np.max(x, 1).reshape(-1, 1))
    res = e_x / e_x.sum(axis=1).reshape(-1, 1)
    return res.reshape(xx.shape)


def predictions_to_json(task, verb_scores, noun_scores, action_scores, action_ids, a_to_vn, top_actions=100, version='0.1', sls=None):
    """Save verb, noun and action predictions to json for submitting them to the EPIC-Kitchens leaderboard"""

    predictions = {'version': version, 'challenge': task, 'results': {}}

    if sls is not None:
        if task == 'action_anticipation':
            predictions['sls_pt'] = 1
            predictions['sls_tl'] = 4
            predictions['sls_td'] = 4
        elif task == 'action_recognition':
            predictions['sls_pt'] = 1
            predictions['sls_tl'] = 4
            predictions['sls_td'] = 4

    row_idxs = np.argsort(action_scores)[:, ::-1]
    top_100_idxs = row_idxs[:, :top_actions]

    action_scores = action_scores[np.arange(
        len(action_scores)).reshape(-1, 1), top_100_idxs]

    for i, v, n, a, ai in zip(action_ids, verb_scores, noun_scores, action_scores, top_100_idxs):
        predictions['results'][str(i)] = {}
        predictions['results'][str(i)]['verb'] = {str(ii): float(vv) for ii, vv in enumerate(v)}
        predictions['results'][str(i)]['noun'] = {str(ii): float(nn) for ii, nn in enumerate(n)}
        predictions['results'][str(i)]['action'] = {"%d,%d" % a_to_vn[ii]: float(aa) for ii, aa in zip(ai, a)}
    return predictions
