#python implementation of autoregression AR for time series prediction, just using OLS
import numpy as np
from test_data import generate_test
from sklearn import linear_model
import numpy.linalg as la
import pandas as pd
import matplotlib.pyplot as plt

"""
takes in time series of dimension p and length n. as n x p matrix.
outputs outlier score, from prediciton error.
"""

#
# def get_seq(X, tw):
#     """
#     returns single or multidimensional set of time sequences length tw
#     from X (vector or matrix n by p), X is np array
#     needs to pad so zeros at start.
#     """
#     n, p = X.shape
#     seqs = []
#     tars = []
#     ind = -tw
#     print('tw={}, n={}, p={}'.format(tw, n, p))
#     while ind +tw <n:
#         start = ind
#         end = ind + tw
#         if ind <0:
#             pad_n = 0-ind
#             pad = np.zeros((pad_n, p))
#             seq = X[:ind+tw,:]
#             seq = np.concatenate([pad,seq])
#         else:
#             seq = X[ind:ind+tw,:]
#         seqs.append(seq)
#         tars.append(X[ind+tw,:]) #target is next one in line.
#         print("ind={}, start={}, end={}, seq shape={}, target ind={}".format(ind, start, end, seq.shape, ind+tw) )
#         ind+=1
#     return seqs, tars
#
#
# def compile_data(data, tw, test_split = 0.2):
#     """
#     assumes data is n,p numpy matrix.
#     """
#     n,p = data.shape
#
#     # Scaling the input data
#     scaler = MinMaxScaler()
#     data = scaler.fit_transform(data)
#     inputs = np.zeros((n-tw,tw,p))
#     targets = np.zeros((n-tw, p))
#
#     for i in range(tw, n):
#         inputs[i-tw] = data[i-tw:i, :]
#         targets[i-tw] = data[i,:]
#     inputs = inputs.reshape(-1,tw,p)
#     targets = targets.reshape(-1,p)
#
#     # Split data into train/test portions and combining all data from different files into a single array
#     test_ind = int(test_split*len(inputs))
#
#     train_x = inputs[:-test_ind, :, :]
#     train_y = targets[:-test_ind, :]
#
#     test_x = inputs[-test_ind:]
#     test_y = targets[-test_ind:]
#
#     return scaler, train_x, train_y, test_x, test_y

def ese(pred, target):
    """
    takes in predicted values and actual values, returns elementwise squared error
    via (x-y)^2
    """
    errs = np.sum((pred - target)**2, axis=0)
    return errs

# def var_model(X, tw):
#     """
#     takes in data, builds model and returns model.
#     trains on using previous tw samples to predict next one.
#     does it per dimension and sums errors.
#     """
#
#     scaler, train_x, train_y, test_x, test_y = compile_data(X, tw)
#     #matrix Y = BZ + U
#     #where Y =
#     return ese(pred, targs)




#
def get_Y (T, p, X):
    Y = []
    for t in range(p-1,T+p-1):
#         print(t)
        Y.append(X[t])
    return np.array(Y).T


def get_Z(T,p,X):
    Z = []
    for t in range(p-1,T+p-1):
#         print(t)
        zt = np.concatenate([[1],np.hstack([X[x].T for x in range(t,t-p,-1)] ) ])
        Z.append(zt)
    Z = np.array(Z)
    return Z.T

def split(p, X):
    """
    p is the number of previous examples we will use to predict the next
    """
    T = X.shape[0]-p #number of data points we have enough data for
    Y = []
    Z = []
    for t in range(T):
        Y.append(X[t+p,:])
        Z.append(np.concatenate([[1],np.hstack([X[x].T for x in range(t+p-1,t-1,-1)] ) ]))
    return np.array(Y).T, np.array(Z).T#, b_hat

def get_estimate(p, X):
    T = X.shape[0]-p #number of data points we have enough data for
    Y,Z = split(p,X)
    # Y = get_Y(T, p, X)
    # Z = get_Z(T, p, X)
    B_hat = Y.dot(Z.T).dot(la.inv(Z.dot(Z.T)))
    return Y, B_hat.dot(Z)

def var_ridge_os(X):
    p = 3
    T = X.shape[0]-p
    Y = get_Y(T, p, X).T
    Z = get_Z(T, p, X).T
    # print(Y.shape)
    # print(Z.shape)
    # Y,Z = split(p,X)
    reg = linear_model.Ridge()
    reg.fit(Z, Y)
    Y_hat = reg.predict(Z)
    errs = ese(Y, Y_hat)
    errs = np.concatenate([np.ones(p)*errs[0], errs])
    return errs


def get_VAR_OS(X):
    """
    use VAR to make estimate then calculate elementwise squared error for each.
    can't do initial p data points as not enough data. they are set to the same
    as the first value.
    """
    p = 3
    Y, Y_hat = get_estimate(p,X)
    errs = ese(Y, Y_hat)
    errs = np.concatenate([np.ones(p)*errs[0], errs])

    return errs

def auc(est_out_scores, outs):
    """
    measures how good the separation is between outliers and inliers
    uses auc
    uses the estimated outlier score from each algorithm.
    """
    from sklearn import metrics
    n = len(est_out_scores)

    actual_os = [1 if i in outs else 0 for i in range(n)]
    try:
        fpr, tpr, thresholds = metrics.roc_curve(actual_os, est_out_scores)
    except:
        print(actual_os[:10], est_out_scores[:10])
        # print(metrics.auc(fpr, tpr))
        raise
    return metrics.auc(fpr, tpr)

if __name__ == '__main__':


    n = 80
    p = 2
    p_lst = [2,4,8,16,32,64]
    gamma = 0.05
    p_frac = 0.3
    p_quant = 0.3
    r = 20
    aucs=[]
    for p in p_lst:

        X, outs = generate_test(n,p,r, p_frac, p_quant,gamma)
        os = get_VAR_OS(X)
        aucs.append(auc(os, outs))
    for i in range(len(p_lst)):
        print(p_lst[i], aucs[i])




    eps = get_VAR_OS(X)
    # T = Y_hat.shape[1]
    tim = np.arange(len(eps))
    fig = plt.figure()
    plt.plot(tim, eps)
    for out in outs:
        fig.text(out+1,2.5, '-', color='r')
        plt.plot(out, eps[out], 'ro')
        print(out)
    plt.show()


    eps = var_ridge_os(X)
    # T = Y_hat.shape[1]
    tim = np.arange(len(eps))
    fig = plt.figure()
    plt.plot(tim, eps)
    for out in outs:
        fig.text(out+1,2.5, '-', color='r')
        plt.plot(out, eps[out], 'ro')
        print(out)
    plt.show()
