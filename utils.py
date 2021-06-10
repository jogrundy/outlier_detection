# utils for outlier detection.

# needs:
import numpy as np
from sklearn import metrics

class TimeoutException(Exception):
    def __init__(self, time):
        Exception.__init__(self, 'timeout after {}s'.format(time))

def ese(pred, target):
    """
    takes in predicted values and actual values, returns elementwise squared error
    via (x-y)^2
    """
    # print(pred.shape, target.shape)
    errs = (pred - target)**2
    # errs = np.sum((pred - target)**2, axis=0)
    # print(errs.shape)
    return errs

def auc(est_out_scores, outs):
    """
    measures how good the separation is between outliers and inliers
    uses auc on the estimated outlier score from each algorithm, compared to
    the actual where 1 is for outlier and 0 is for not.
    """
    n = len(est_out_scores)
    actual_os = [1 if i in outs else 0 for i in range(n)]
    fpr, tpr, thresholds = metrics.roc_curve(actual_os, est_out_scores)
    return metrics.auc(fpr, tpr)


def fps(est_out_scores, outs):
    """
    measures how good the separation is between outliers and inliers
    uses number of false positives found after finding all outliers
    uses the estimated outlier score from each algorithm.
    higher score = more outlier
    """
    inds = np.flip(np.argsort(est_out_scores)) #gives indices in size order
    n = len(est_out_scores)
    for i in range(n):
        if len(np.setdiff1d(outs,inds[:i]))==0: #everything in outs is also in inds
            fps = len(np.setdiff1d(inds[:i], outs)) #count the things in inds not in outs
            return fps/i
    return 1
