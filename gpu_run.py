import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from test_data import generate_test
from time import time
from sklearn.svm import OneClassSVM
from admm_graph_OP_tweak import GOP
from Simple_GOP import SGOP
from outlier_pursuit import outlier_pursuit
from ae import get_ae_losses
from vae import get_vae_losses
from sklearn.ensemble import IsolationForest
from gru import get_GRU_os, get_LSTM_os
from sklearn import metrics
from sklearn.cluster import DBSCAN
from sklearn.mixture import GaussianMixture
from var import get_VAR_OS
import os
import stopit
import datetime
plt.rcParams.update({'font.size': 18})

class TimeoutException(Exception):
    def __init__(self, time):
        Exception.__init__(self, 'timeout after {}s'.format(time))


def ese(pred, target):
    """
    takes in predicted values and actual values, returns elementwise squared error
    via (x-y)^2
    """
    errs = (pred - target)**2
    return errs

def OLS_err(X_train, y_train, X, y):
    """
    takes in train test split returns elementwise error for whole dataset.
    """
    reg = linear_model.LinearRegression()
    reg.fit(X_train, y_train)
    pred = reg.predict(X)
    return ese(pred, y)

def ridge_err(X_train, y_train, X, y):
    """
    takes in train test split returns elementwise error for whole dataset.
    """
    reg = linear_model.Ridge()
    reg.fit(X_train, y_train)
    pred = reg.predict(X)
    return ese(pred, y)

def lasso_err(X_train, y_train, X, y):
    """
    takes in train test split returns elementwise error for whole dataset.
    """
    reg = linear_model.Lasso()
    reg.fit(X_train, y_train)
    pred = reg.predict(X)
    return ese(pred, y)

def get_reg_os(X):
    n,p = X.shape
    err_sum = np.zeros(n)

    for i in range(p):
        inds = np.arange(p)
        inds = inds
        X_x = np.delete(X, i, axis=1)
        y_y = X[:,i]
        X_train, X_test, y_train, y_test = train_test_split(X_x, y_y)
        err = OLS_err(X_train, y_train, X_x, y_y)
        err_sum +=err
    return err_sum/n

def get_ridge_os(X):
    n,p = X.shape
    err_sum = np.zeros(n)

    for i in range(p):
        inds = np.arange(p)
        inds = inds
        X_x = np.delete(X, i, axis=1)
        y_y = X[:,i]
        X_train, X_test, y_train, y_test = train_test_split(X_x, y_y)
        err = ridge_err(X_train, y_train, X_x, y_y)
        err_sum +=err
    return err_sum/n

def get_LASSO_os(X):
    n,p = X.shape
    err_sum = np.zeros(n)

    for i in range(p):
        inds = np.arange(p)
        inds = inds
        X_x = np.delete(X, i, axis=1)
        y_y = X[:,i]
        X_train, X_test, y_train, y_test = train_test_split(X_x, y_y)
        err = lasso_err(X_train, y_train, X_x, y_y)
        err_sum +=err
    return err_sum/n

# The testing algorithms


#regression
def test_VAR(X):
    os = get_VAR_OS(X)
    return os


def test_OLS(X):
    """
    takes in only data 'X', in samples as rows format
    returns only list of outlier scores for each sample
    higher score = more outlier
    """
    losses = get_reg_os(X)
    # print(len(losses))
    #loss here is summed elementwise errors
    return losses

def test_Ridge(X):
    """
    takes in only data 'X', in samples as rows format
    returns only list of outlier scores for each sample
    higher score = more outlier
    """
    losses = get_ridge_os(X)
    # print(len(losses))
    #loss here is summed elementwise errors
    return losses

def test_LASSO(X):
    """
    takes in only data 'X', in samples as rows format
    returns only list of outlier scores for each sample
    higher score = more outlier
    """
    losses = get_LASSO_os(X)
    # print(len(losses))
    #loss here is summed elementwise errors
    return losses

#tersting algorithms
#density

def test_OCSVM(X):
    """
    takes in only data 'X'
    returns only list of outlier scores for each sample
    higher score = more outlier
    """
    clf = OneClassSVM(gamma='scale')
    clf.fit(X)
    dists = clf.decision_function(X)*-1
    return dists #largest is now most outlier

def test_GMM(X):
    """
    takes in only data 'X', in samples as rows format
    returns only list of outlier scores for each sample
    higher score = more outlier
    """
    k = 3
    # arr, pi_mu_sigs,i = em(X, k, 1000)
    # log_likelihoods = log_Ls(X, pi_mu_sigs)
    clf = GaussianMixture(n_components=k)
    clf.fit(X)
    scores = clf.score_samples(X)*-1 # returns log probs for data
    return scores #to give in higher score = more outlier

def test_IF(X):
    clf = IsolationForest()#contamination='auto', behaviour='new')
    clf.fit(X)
    os = clf.decision_function(X)
    return os*-1 # average number splits to isolation. small is outlier.

def test_DBSCAN(X):
    """
    takes in only data 'X', in samples as rows format
    DBSCAN from sklearn returns -1 as label for outliers.
    use from scartch implementaiton and get distance from nn as os
    returns only list of outlier scores for each sample
    higher score = more outlier
    own implementation is very slow for higher N..
    """
    n,p = X.shape
    eps = 0.3 #normalised data
    if int(n//20) < 3:
        minnum = 3
    elif int(n//20) > 100:
        minnum = 100
    else:
        minnum = int(n//20)
    # point_classes, cl, os = dbscan(X, eps, minnum)
    clf = DBSCAN(eps=eps, min_samples=minnum)
    classes = clf.fit_predict(X)
    # print(classes)
    #returns only in class or out of class binary classification
    i = -1
    n_found = 0
    cl_sizes = {}
    while n_found <n:

        n_found_inds = len(np.where(classes == i)[0])
        n_found += n_found_inds
        # print(i, n_found_inds)
        cl_sizes[i] = n_found_inds
        i+=1
    # print(cl_sizes)
    cl_lst = [i[0] for i in sorted(cl_sizes.items(), key=lambda k:k[1], reverse=True)]
    # print(cl_lst)
    n_classes = len(cl_lst)

    # most populous group get score zero, then 1, 2, etc..
    os = [n_classes if x<0 else x for x in classes]
    # print(os)

    # raise
    # os = [1 if x < 0 else 0 for x in classes]
    return np.array(os)

# deep learning algorithms
def test_VAE(X):
    """
    takes in only data 'X', in samples as rows format
    returns only list of outlier scores for each sample
    higher score = more outlier
    """
    losses = get_vae_losses(X)
    # print(losses[:10])
    #gives reconstruciton error from AE, should be largest for outliers
    return losses

def test_AE(X):
    """
    takes in only data 'X', in samples as rows format
    returns only list of outlier scores for each sample
    higher score = more outlier
    """
    losses = get_ae_losses(X)
    #gives reconstruciton error from AE, should be largest for outliers
    return losses

def test_GRU(X):
    """
    takes in only data 'X', in samples as rows format
    returns only list of outlier scores for each sample
    higher score = more outlier
    """
    errs = get_GRU_os(X)
    #gives error from GRU, should be largest for outliers
    return errs

def test_LSTM(X):
    """
    takes in only data 'X', in samples as rows format
    returns only list of outlier scores for each sample
    higher score = more outlier
    """
    errs = get_LSTM_os(X)
    #gives error from LSTM, should be largest for outliers
    errs = np.array(errs).reshape(-1)
    return errs


# Matrix methods

def test_OP(X):
    """
    takes in only data 'X', in samples as rows format
    returns only list of outlier scores for each sample
    higher score = more outlier
    """
    lamb = 0.5
    M = X.T
    L_hat, C_hat, count = outlier_pursuit(M, lamb)
    return np.sum(C_hat, axis=0)

def test_GOP(X):
    """
    takes in only data 'X', in samples as rows format
    returns only list of outlier scores for each sample
    higher score = more outlier
    """
    lamb = 0.5
    gamma = 0.1
    M = X.T
    S_hat = GOP(M, lamb, gamma)
    return np.sum(S_hat, axis=0)

def test_SGOP(X):
    """
    takes in only data 'X', in samples as rows format
    returns only list of outlier scores for each sample
    higher score = more outlier
    """
    lamb = 0.5
    gamma = 0.1
    M = X.T
    S_hat = SGOP(M, lamb, gamma)
    return np.sum(S_hat, axis=0)

# end of testing algorithms

def test_algo(X, outs, algo, metric):
    """
    takes in algorithm 'algo', data 'X', with outlier indices 'outs'
    returns fp rate, as given by separation_metric
    algo must have input only X
    """
    outlier_scores = algo(X)
    fps = metric[1](outlier_scores, outs)
    aucs = metric[0](outlier_scores, outs)

    return fps, aucs

def contour_fp_algo(n, p, r, ta, n_steps, n_runs, gamma, algo, metric):
    """
    does 2d contour plot varying
    p_frac - number of parameters changed in the outliers
    p_quant - amount each parameter is varied by
    ie when both are 0, there are no outliers in terms of data
    """
    # step_size = 1/n_steps
    pf = np.linspace(0,1,n_steps)
    pq = np.linspace(0,1,n_steps)

    fps = []
    for p_frac in pf:
        # print(p_frac)
        fp_row=[]
        for p_quant in pq:
            # print(p_quant)
            runs=[]
            for i in range(n_runs):

                # print('run {}'.format(i))
                # print(n,p,r)
                la_err = False
                while not la_err:
                    try:
                        X, outs = generate_test(n, p, r, p_frac, p_quant, gamma, ta)
                        fp, auc = test_algo(X, outs, algo, metric)
                        la_err = False
                    except numpy.linalg.LinAlgError as err:
                        if 'Singular matrix' in str(err):
                            la_err = True
                            print('redoing due to sungular matrix err')
                runs.append(fp)
            # print(runs)
            fp_row.append(np.mean(runs))
        fps.append(fp_row)

    fpz = np.array(fps)
    # print(fps)

    return pf, pq, fpz

def auc(est_out_scores, outs):
    """
    measures how good the separation is between outliers and inliers
    uses auc
    uses the estimated outlier score from each algorithm.
    """
    n = len(est_out_scores)

    actual_os = [1 if i in outs else 0 for i in range(n)]
    try:
        fpr, tpr, thresholds = metrics.roc_curve(actual_os, est_out_scores)
    except:
        print(actual_os[:10], est_out_scores[:10])
        # print(metrics.auc(fpr, tpr))
        raise
    return metrics.auc(fpr, tpr)




def separation_metric(est_out_scores, outs):
    """
    measures how good the separation is between outliers and inliers
    uses number of false positives found after finding all outliers
    uses the estimated outlier score from each algorithm.
    higher score = more outlier
    """
    # n = len(est_out_scores)
    #
    # actual_os = [1 if i in outs else 0 for i in range(n)]
    # fpr, tpr, thresholds = metrics.roc_curve(actual_os, est_out_scores)
    # print(fpr)
    # # print(fpr, tpr, thresholds)
    # # print(metrics.auc(fpr, tpr))
    # return fpr

    inds = np.flip(np.argsort(est_out_scores)) #gives indices in size order
    n = len(est_out_scores)
    for i in range(n):
#         print(inds[:i])
#         print(outs)
        if len(np.setdiff1d(outs,inds[:i]))==0: #everything in outs is also in inds
            fps = len(np.setdiff1d(inds[:i], outs)) #count the things in inds not in outs
            return fps/i
    return 1

def plot_each_algo_for_each_ta(n, p, r, ta_lst, n_steps, n_runs, algo_list, str_algo_lst):

    w = len(algo_list)
    v = len(ta_lst)
    t0 = time()
    plt.figure(figsize=(10,10))
    for j in range(v):
        for i in range(w):
            t1 = time()
            algo = algo_list[i]
            # ta = ta_lst[j]
            print('{}'.format(str_algo_lst[i]))

            pf, pq, fpz = contour_fp_algo(n, p, r, j+1, n_steps, n_runs, gamma,algo)
            zs = np.round(100*(fpz.size-np.count_nonzero(fpz))/fpz.size) # gives fraction of zeros

            plt.subplot(v,w, (w*j + i)+1)
            label = 'ta_{}'.format(j+1)
            plt.title('FPs avg. {} runs for {}'.format(n_runs, label))
            plt.contourf(pf, pq, fpz)
            plt.colorbar()
            plt.xlabel('p_frac')
            plt.ylabel('p_quant')

            plt.annotate(str_algo_lst[i], (0.8,0.9))
            plt.annotate('{}% zero'.format(int(zs)), (0.8,0.8))
            t2 = time()-t1
            print('Algorithm {} with data {} took {}m and {}s to run {} times '.format(str_algo_lst[i],
                                                                        label,
                                                                        int(t2//60),
                                                                        int(t2%60),
                                                                        n_steps*n_runs*n_steps))


    t3 = time()-t0
    print('Took {}m {}s to run all algorithms'.format(int(t3//60),int(t3%60)))
    fname = './images/test_n{}_p{}_r{}_FPta_plot.eps'.format(n,p,r)
    plt.savefig(fname, bbox_inches='tight', pad_inches=0)

    plt.show()

def contour_auc_pfq(pf_lst, pq_lst, r, noise, ta, n, p, n_runs, gamma,algo, algo_str,metric, timeout, outlier_type):
    """
    does 2d contour plot
    varying p frac and p_quant, using ceiling, so always at least 1 outlier
    """

    all_name = './results/{}_pfq_all.txt'.format(timestamp)
    if not os.path.isfile(all_name):
        with open(all_name, 'w') as f:
            info = '{}, pfq,{},runs={},n={},p={},ta={}\n'.format(timestamp, outlier_type,
                                                        n_runs,n,p,ta)
            f.write(info)
    fps = []
    aucs = []
    for p_frac in pf_lst:
        # print(p_frac)
        fp_row=[]
        auc_row=[]
        succeed = True
        for p_quant in pq_lst:
            Fail = False
            t0 = time()

            # print(p_quant)
            fp_runs=[]
            auc_runs=[]
            # n = 10**n_pow
            # p = 2**p_pow

            for i in range(n_runs):
                la_err = True
                while la_err and succeed:
                    try:
                        X, outs = generate_test(n, p, r, p_frac, p_quant, gamma, noise, ta=ta, nz_cols=None, outlier_type=outlier_type)
                        with stopit.ThreadingTimeout(timeout) as ctx_mgr:
                            fp, auc = test_algo(X, outs, algo, metric)
                        if ctx_mgr.state==ctx_mgr.TIMED_OUT:
                            raise TimeoutException(timeout)
                        la_err = False
                        # print('got to end of try')

                    except np.linalg.LinAlgError as err:
                        if 'Singular matrix' in str(err):
                            la_err = True
                            print('redoing due to singular matrix err')
                        else:
                            # print(err)
                            print('some other linalg error')
                            raise(err)
                    except TimeoutException as err:
                        # print('timeout after {}s'.format(timeout))
                        succeed = False
                        #want it not to bother to run another run,
                        #and not to bother trying the next n_pow up
                        # raise(err)
                if succeed:
                    fp_runs.append(fp)
                    auc_runs.append(auc)
                else:
                    break
            t1 = time() - t0
            if Fail:
                Fail = False
                fp_row.append(np.nan)
                auc_row.append(np.nan)
                print('n={}, p={}, Failed, LinAlgError'.format(n, p))
            elif not succeed:
                print('n={}, p={}, Failed, Timeout after {}s'.format(n, p, timeout))
                fp_row.append(np.nan)
                auc_row.append(np.nan)
                with open(all_name, 'a') as f:
                    fp_str = '{}, {}, {}, {}, {}, {}\n'.format(algo_str, ta, 'fps',n,p, np.nan)
                    auc_str = '{}, {}, {}, {}, {}, {}\n'.format(algo_str, ta, 'auc',n,p, np.nan)
                    f.write(fp_str)
                    f.write(auc_str)
            else:
                # print(runs)
                fp_row.append(np.mean(fp_runs))
                auc_row.append(np.mean(auc_runs))
                #saving raw data to file
                with open(all_name, 'a') as f:
                    fp_str = '{}, {}, {}, {}, {}, '.format(algo_str, ta, 'fps',p_frac,p_quant)
                    fp_str = fp_str+''.join(['%0.3f, '])*len(fp_runs)%tuple(fp_runs)+'\n'
                    auc_str = '{}, {}, {}, {}, {}, '.format(algo_str, ta, 'auc',p_frac,p_quant)
                    auc_str = auc_str+''.join(['%0.3f, '])*len(auc_runs)%tuple(auc_runs)+'\n'
                    f.write(fp_str)
                    f.write(auc_str)
                print('p_frac={}, quant={}, runs={}, time= {}m {}s'.format(round(p_frac,3), round(p_quant,3), n_runs, int(t1//60),int(t1%60)))
        fps.append(fp_row)
        aucs.append(auc_row)

    fpz = np.array(fps)
    aucz = np.array(aucs)
    # print(fps)

    return fpz, aucz

def get_auc_noise(p_frac, p_quant, r, noise_list, ta, n, p, n_runs, gamma,algo, metric, timeout, outlier_type):
    """
    runs each algorithm with varying amounts of noise on ta given.
    """
    all_name = './results/{}_noise_all.txt'.format(timestamp) #lazy programming using global
    if not os.path.isfile(all_name):
        with open(all_name, 'w') as f:
            info = '{}, {}, {}, {}, {}, {}, '.format('algo','ta', 'n', 'p','fps', 'noise')
            # print(len(np.arange(n_runs)), n_runs)
            info2 = ''.join(['%d, '])*n_runs%tuple(np.arange(n_runs)+1)
            # format([np.arange(n_runs)+1])
            # print(info, info2)
            f.write(info+info2[:-2]+'\n')
    # print(info+info2[:-2]+'\n')
    # raise
    fps = []
    aucs = []
    for noise in noise_list:
        Fail = False
        t0 = time()
        fp_runs=[]
        auc_runs=[]
        succeed=True

        for i in range(n_runs):
            la_err = True
            while la_err and succeed:
                try:
                    X, outs = generate_test(n, p, r, p_frac, p_quant, gamma, noise, ta=ta, nz_cols=None, outlier_type=outlier_type)
                    with stopit.ThreadingTimeout(timeout) as ctx_mgr:
                        fp, auc = test_algo(X, outs, algo, metric)
                    if ctx_mgr.state==ctx_mgr.TIMED_OUT:
                        raise TimeoutException(timeout)
                    la_err = False
                    # print('got to end of try')

                except np.linalg.LinAlgError as err:
                    if 'Singular matrix' in str(err):
                        la_err = True
                        print('redoing due to singular matrix err')
                    else:
                        # print(err)
                        print('some other linalg error')
                        raise(err)
                except TimeoutException as err:
                    # print('timeout after {}s'.format(timeout))
                    succeed = False

            if succeed:
                fp_runs.append(fp)
                auc_runs.append(auc)

            else:
                break
        t1 = time() - t0
        if Fail:
            Fail = False
            fp_row.append(np.nan)
            auc_row.append(np.nan)
            print('n={}, p={}, Failed, LinAlgError'.format(n, p))
        elif not succeed:
            print('n={}, p={}, Failed, Timeout after {}s'.format(n, p, timeout))
            fp_row.append(np.nan)
            auc_row.append(np.nan)
            with open(all_name, 'a') as f:
                fp_str = '{}, {}, {}, {}, {}, {}\n'.format(algo_str, ta, 'fps',n,p, np.nan)
                auc_str = '{}, {}, {}, {}, {}, {}\n'.format(algo_str, ta, 'auc',n,p, np.nan)
                f.write(fp_str)
                f.write(auc_str)
        else:
            # print(runs)
            fps.append(np.mean(fp_runs))
            aucs.append(np.mean(auc_runs))
            with open(all_name, 'a') as f:
                fp_str = '{}, {}, {}, {}, {}, {}, '.format(algo, ta, n,p,'fps',noise)
                fp_str = fp_str+''.join(['%0.3f, '])*len(fp_runs)%tuple(fp_runs)+'\n'
                auc_str = '{}, {}, {}, {}, {}, {}, '.format(algo, ta, n,p,'auc',noise)
                auc_str = auc_str+''.join(['%0.3f, '])*len(auc_runs)%tuple(auc_runs)+'\n'
                f.write(fp_str)
                f.write(auc_str)
            # print('p_frac={}, quant={}, runs={}, time= {}m {}s'.format(round(p_frac,3), round(p_quant,3), n_runs, int(t1//60),int(t1%60)))


            print('noise={}, runs={}, time= {}m {}s'.format(noise, n_runs, int(t1//60),int(t1%60)))
    # fps.append(fp_row)
    # aucs.append(auc_row)

    fpz = np.array(fps)
    aucz = np.array(aucs)
    # print(fps)

    return fpz, aucz

def contour_fp_np(n_lst, p_lst, r, noise, ta, p_quant, p_frac, n_runs, gamma,algo,algo_str, metric, timeout, nz_cols, outlier_type):
    """
    does 2d contour plot varying
    n - number of samples
    p - number of features
    with 0.2 p frac and p_quant, using ceiling, so always at least 1 outlier
    """
    # step_size = 1/n_steps
    # p_quant = 0.2
    # p_frac = 0.2
    all_name = './results/{}_np_all.txt'.format(timestamp)

    if not os.path.isfile(all_name):
        with open(all_name, 'w') as f:
            info = '{}, np,{},runs={},p_frac={},p_quant={},ta={}\n'.format(timestamp, outlier_type,
                                                        n_runs,p_frac,p_quant,ta)
            f.write(info)

    fps = []
    aucs = []
    for p_pow in p_lst:
        # print(p_frac)
        fp_row=[]
        auc_row=[]
        succeed = True
        for n_pow in n_lst:
            Fail = False
            t0 = time()

            # print(p_quant)
            fp_runs=[]
            auc_runs=[]
            n = 10**n_pow
            p = 2**p_pow

            for i in range(n_runs):

                la_err = True
                while la_err and succeed:
                    try:
                        X, outs = generate_test(n, p, r, p_frac, p_quant, gamma, noise, ta=ta, nz_cols=nz_cols, outlier_type=outlier_type)
                        with stopit.ThreadingTimeout(timeout) as ctx_mgr:
                            fp, auc = test_algo(X, outs, algo, metric)
                        if ctx_mgr.state==ctx_mgr.TIMED_OUT:
                            raise TimeoutException(timeout)
                        la_err = False
                        # print('got to end of try')

                    except np.linalg.LinAlgError as err:
                        if 'Singular matrix' in str(err):
                            la_err = True
                            print('redoing due to singular matrix err')
                        elif 'SVD did not converge':
                            la_err = True
                            print('redoing due to SVD not converging')
                        else:
                            # print(err)
                            print('some other linalg error')
                            raise(err)
                    except TimeoutException as err:
                        # print('timeout after {}s'.format(timeout))
                        succeed = False
                        #want it not to bother to run another run,
                        #and not to bother trying the next n_pow up
                        # raise(err)
                if succeed:
                    fp_runs.append(fp)
                    auc_runs.append(auc)
                else:
                    break
            t1 = time() - t0
            if Fail:
                Fail = False
                fp_row.append(np.nan)
                auc_row.append(np.nan)
                print('n={}, p={}, Failed, LinAlgError'.format(n, p))
            elif not succeed:
                print('n={}, p={}, Failed, Timeout after {}s'.format(n, p, timeout))
                fp_row.append(np.nan)
                auc_row.append(np.nan)
                with open(all_name, 'a') as f:
                    fp_str = '{}, {}, {}, {}, {}, {}\n'.format(algo_str, ta, 'fps',n,p, np.nan)
                    auc_str = '{}, {}, {}, {}, {}, {}\n'.format(algo_str, ta, 'auc',n,p, np.nan)
                    f.write(fp_str)
                    f.write(auc_str)
            else:
                # print(runs)
                fp_row.append(np.mean(fp_runs))
                auc_row.append(np.mean(auc_runs))
                #saving raw data to file
                with open(all_name, 'a') as f:
                    fp_str = '{}, {}, {}, {}, {}, '.format(algo_str, ta, 'fps',n,p)
                    fp_str = fp_str+''.join(['%0.3f, '])*len(fp_runs)%tuple(fp_runs)+'\n'
                    auc_str = '{}, {}, {}, {}, {}, '.format(algo_str, ta, 'auc',n,p)
                    auc_str = auc_str+''.join(['%0.3f, '])*len(auc_runs)%tuple(auc_runs)+'\n'
                    f.write(fp_str)
                    f.write(auc_str)
                print('n={}, p={}, runs={}, time= {}m {}s'.format(n, p, n_runs, int(t1//60),int(t1%60)))
        fps.append(fp_row)
        aucs.append(auc_row)

    fpz = np.array(fps)
    aucz = np.array(aucs)
    # print(fps)

    return fpz, aucz

def plot_each_algo_for_pfq(pf_lst, pq_lst, r, gamma, noise, ta, n, p, n_runs,
                            algo_list, algo_type, metric, metric_str,
                            timeout, timestamp, outlier_type):

    w = len(algo_list)
    v = 1

    t0 = time()
    fig_size_x = len(algo_list)*5
    fig_size_y = 5
    plt.figure(figsize=(fig_size_x, fig_size_y))
    ts = []
    # fp_score = []
    # auc_score = []
    for i in range(w):
        # To Do: code up keeping the colour bar max and min constant, 0 to 1 - done
        t1 = time()
        algo_str = algo_lst[i]
        algo = algo_dict[algo_str]
        print('{}'.format(algo_str))

        fpz, aucz = contour_auc_pfq(pf_lst, pq_lst, r, noise, ta, n, p, n_runs, gamma,algo,algo_str, metric_lst, timeout, outlier_type)
        plt.subplot(v,w, (w*0 + i)+1)
        label = 'ta_{}'.format(ta)
        plt.title('{}'.format(algo_str))
        plt.contourf(pf_lst, pq_lst, aucz, np.arange(0,1+1e-8,0.05),vmin=0, vmax=1)
        if i == w-1:
            plt.colorbar()
        plt.xlabel (r'p frac')
        plt.ylabel(r'p quant')
        t2 = time()-t1
        # plt.annotate('{}m {}s'.format(int(t2//60),int(t2%60)), (0.8,0.8) )
        ts.append(t2)
        # fp_score.append(fpz)
        # auc_score.append(aucz)

        print('Algorithm {} with data {} took {}m and {}s to run {} times'.format(algo_str,
                                                                    label,
                                                                    int(t2//60),
                                                                    int(t2%60),
                                                                    len(pf_lst)*len(pq_lst)*n_runs))


    t3 = time()-t0

    print('Took {}m {}s to run all {} algorithms'.format(int(t3//60),int(t3%60), algo_type))
    fname = './images/{}_pfq_{}_n_{}_p_{}_ta{}.eps'.format(timestamp,
                                                algo_type, n, p, ta)

    plt.savefig(fname, bbox_inches='tight', pad_inches=0)
    # txt_fname='./results/{}_pfq_results.txt'.format(timestamp)
    #
    # raw_fname='./results/{}_pfq_raw.txt'.format(timestamp)

    # if os.path.isfile(raw_fname):
    #     with open(raw_fname, 'a') as f:
    #         # info = '{}_np_{}_pfrac_{}_pquant_{}_ta{}\n'.format(timestamp,
    #         #                                             algo_type, p_frac, p_quant, ta)
    #         # f.write(info)
    #         for i in range(len(algo_lst)):
    #             fps = fp_score[i].flatten()
    #             fp_str = algo_lst[i]+ ', ' + str(ta) +', ' + outlier_type+', fps, '+''.join(['%0.3f, '])*len(fps)%tuple(fps)
    #             fp_str = fp_str[:-2]+'\n'
    #             f.write(fp_str)
    #             aucs = auc_score[i].flatten()
    #             auc_str = algo_lst[i]+', ' + str(ta) +', ' + outlier_type+', auc, '+ ''.join(['%0.3f, '])*len(aucs)%tuple(aucs)
    #             auc_str = auc_str[:-2]+'\n'
    #             f.write(auc_str)
    # else:
    #     with open(raw_fname, 'w') as f:
    #         info = '{}_pfq_{}_{}_n_{}_p_{}_ta{}\n'.format(timestamp, outlier_type,
    #                                                     algo_type, n, p, ta)
    #         f.write(info)
    #         for i in range(len(algo_lst)):
    #             fps = fp_score[i].flatten()
    #             fp_str = algo_lst[i]+', ' + str(ta) +', ' + outlier_type+', fps, '+''.join(['%0.3f, '])*len(fps)%tuple(fps)
    #             fp_str = fp_str[:-2]+'\n'
    #             f.write(fp_str)
    #             aucs = auc_score[i].flatten()
    #             auc_str = algo_lst[i]+', ' + str(ta) +', ' + outlier_type+', auc, '+ ''.join(['%0.3f, '])*len(aucs)%tuple(aucs)
    #             auc_str = auc_str[:-2]+'\n'
    #             f.write(auc_str)
    #
    # if os.path.isfile(txt_fname):
    #     with open(txt_fname, 'a') as f:
    #         for i in range(len(algo_lst)):
    #             txt = '{}, {}, {}, {}, {}, {}, {},{}, {}, {}, {}, {}, {}\n'.format(timestamp,
    #                                         outlier_type, algo_lst[i], ta, n, p, n_runs,
    #                                         len(pf_lst)*len(pq_lst)*n_runs, int(pf_lst[-1]),
    #                                         int(pf_lst[-1]), int(ts[i]), np.mean(fp_score[i]),
    #                                         np.mean(auc_score[i]))
    #             f.write(txt)
    # else:
    #     with open(txt_fname, 'w') as f:
    #         f.write('algo, ta, outlier_type, p_frac, p_quant, n_runs, total_n_runs, max_n, max_p, total_time, fp_score, auc_score\n')
    #         for i in range(len(algo_lst)):
    #             txt = '{}, {}, {}, {}, {}, {}, {}, {},{}, {}, {}, {}, {}\n'.format(timestamp,
    #                                         outlier_type, algo_lst[i], ta, n, p, n_runs,
    #                                         len(pf_lst)*len(pq_lst)*n_runs, int(pf_lst[-1]),
    #                                         int(pf_lst[-1]), int(ts[i]), np.mean(fp_score[i]),
    #                                         np.mean(auc_score[i]))
    #             f.write(txt)
    #
    #
    #
    # # plt.show()
    plt.close()
def plot_noise(p_frac, p_quant, r, gamma, noise_list, ta, n, p, n_runs,algo_list,algo_type,
                metric, metric_str, timeout, timestamp, outlier_type):

    w = len(algo_list)
    v = 1

    t0 = time()
    fig_size_x = 5
    fig_size_y = 5
    plt.figure(figsize=(fig_size_x, fig_size_y))
    ts = []
    # fp_score = [] #one row of numbers for one al
    # auc_score = []
    plt.title('{} avg. {} runs of {} with outlier type {}'.format(metric_str[0], n_runs, ta, outlier_type))
    for i in range(w):
        # To Do: code up keeping the colour bar max and min constant, 0 to 1 - done
        t1 = time()
        algo_str = algo_lst[i]
        algo = algo_dict[algo_str] #using global naughty..
        print('{}'.format(algo_str))

        fpz, aucz = get_auc_noise(p_frac, p_quant, r, noise_list, ta, n, p, n_runs, gamma,algo, metric_lst, timeout, outlier_type)

        # plt.subplot(v,w, (w*0 + i)+1)

        label = ' ta_{}'.format(ta)
        plt.plot(noise_list, aucz, label=algo_str)#+label)

        plt.xlabel (r'noise')
        plt.ylabel(r'AUC score')
        t2 = time()-t1
        # plt.annotate('{}m {}s'.format(int(t2//60),int(t2%60)), (0.8,0.8) )
        ts.append(t2)
        # fp_score.append(fpz)
        # auc_score.append(aucz)

        print('Algorithm {} with data {} took {}m and {}s to run {} times'.format(algo_str,
                                                                    label,
                                                                    int(t2//60),
                                                                    int(t2%60),
                                                                    len(noise_list)*n_runs))


    t3 = time()-t0
    plt.legend()

    print('Took {}m {}s to run all algorithms on outlier type {}'.format(int(t3//60),int(t3%60), outlier_type))
    fname = './images/{}_noise_{}_{}_n_{}_p_{}_ta{}.eps'.format(timestamp, outlier_type, algo_type, n, p, ta)
    # print(len(fp_score))
    # print(len(auc_score))
    plt.savefig(fname, bbox_inches='tight', pad_inches=0)
    # txt_fname='./results/{}_{}_noise_results.txt'.format(timestamp, algo_type)
    #
    # raw_fname='./results/{}_{}_noise_raw.txt'.format(timestamp, algo_type)
    #
    # if os.path.isfile(raw_fname):
    #     with open(raw_fname, 'a') as f:
    #         # info = '{}_np_{}_pfrac_{}_pquant_{}_ta{}\n'.format(timestamp,
    #         #                                             algo_type, p_frac, p_quant, ta)
    #         # f.write(info)
    #         for i in range(len(algo_lst)):
    #             fps = fp_score[i].flatten()
    #             fp_str = algo_lst[i]+ ', ' + str(ta) +', ' + outlier_type+', fps, '+''.join(['%0.3f, '])*len(fps)%tuple(fps)
    #             fp_str = fp_str[:-2]+'\n'
    #             f.write(fp_str)
    #             aucs = auc_score[i].flatten()
    #             auc_str = algo_lst[i]+', ' + str(ta) +', ' + outlier_type+', auc, '+ ''.join(['%0.3f, '])*len(aucs)%tuple(aucs)
    #             auc_str = auc_str[:-2]+'\n'
    #             f.write(auc_str)
    # else:
    #     with open(raw_fname, 'w') as f:
    #         info = '{}_noise_{}_n_{}_p_{}_ta{}\n'.format(timestamp, outlier_type, n, p, ta)
    #         f.write(info)
    #         for i in range(len(algo_lst)):
    #             fps = fp_score[i].flatten()
    #             fp_str = algo_lst[i]+', ' + str(ta) +', ' + outlier_type+', fps, '+''.join(['%0.3f, '])*len(fps)%tuple(fps)
    #             fp_str = fp_str[:-2]+'\n'
    #             f.write(fp_str)
    #             aucs = auc_score[i].flatten()
    #             auc_str = algo_lst[i]+', ' + str(ta) +', ' + outlier_type+', auc, '+ ''.join(['%0.3f, '])*len(aucs)%tuple(aucs)
    #             auc_str = auc_str[:-2]+'\n'
    #             f.write(auc_str)
    #
    # if os.path.isfile(txt_fname):
    #     with open(txt_fname, 'a') as f:
    #         for i in range(len(algo_lst)):
    #             # print(i)
    #             txt = '{}, {}, {}, {}, {}, {}, {},{}, {}, {}, {}, {}\n'.format(timestamp,
    #                                         outlier_type, algo_lst[i], ta,  n, p, n_runs,
    #                                         len(noise_list)*n_runs, noise_list, int(ts[i]),
    #                                         np.mean(fp_score[i]), np.mean(auc_score[i]))
    #             f.write(txt)
    # else:
    #     with open(txt_fname, 'w') as f:
    #         f.write('timestamp, outlier_type,algo, ta,  p_frac, p_quant, n_runs, total_n_runs, noise_list, total_time, fp_score, auc_score\n')
    #         for i in range(len(algo_lst)):
    #             txt = '{}, {}, {}, {}, {}, {}, {}, {},{}, {}, {}, {}\n'.format(timestamp,
    #                                         outlier_type, algo_lst[i], ta, n, p, n_runs,
    #                                         len(noise_list)*n_runs, noise_list, int(ts[i]),
    #                                         np.mean(fp_score[i]), np.mean(auc_score[i]))
    #             f.write(txt)



    # plt.show()
    plt.close()

def plot_each_algo_for_np(n_lst, p_lst, r, gamma, noise, ta, p_quant, p_frac,n_runs,
                            algo_list, algo_type, metric, metric_str, timeout,
                            timestamp, nz_cols, outlier_type):

    w = len(algo_list)
    v = 1

    t0 = time()
    fig_size_x = len(algo_list)*5
    fig_size_y = 5
    plt.figure(figsize=(fig_size_x, fig_size_y))
    ts = []
    # fp_score = []
    # auc_score = []
    for i in range(w):
        # To Do: code up keeping the colour bar max and min constant, 0 to 1

        t1 = time()
        algo_str = algo_lst[i]
        algo = algo_dict[algo_str]


        print('{}'.format(algo_str))

        fpz, aucz = contour_fp_np(n_lst, p_lst, r, noise, ta,p_quant, p_frac, n_runs, gamma,algo, algo_str, metric_lst, timeout, nz_cols, outlier_type)
        # zs = np.round(100*(fpz.size-np.count_nonzero(fpz))/fpz.size) # gives fraction of zeros

        plt.subplot(v,w, (w*0 + i)+1)
        label = 'ta_{}'.format(ta)
        plt.title('{} avg. {} runs of {}'.format(metric_str[0], n_runs, algo_str))
        plt.contourf(n_lst, p_lst, aucz, np.arange(0,1+1e-8,0.05),vmin=0, vmax=1)
        if i == w-1:
            plt.colorbar()
        plt.xlabel (r'10^n samples')
        plt.ylabel(r'2^p features')
        t2 = time()-t1
        # plt.annotate('{}m {}s'.format(int(t2//60),int(t2%60)), (0.8,0.8) )
        ts.append(t2)
        # fp_score.append(fpz)
        # auc_score.append(aucz)

        print('Algorithm {} with data {} took {}m and {}s to run {} times'.format(algo_str,
                                                                    label,
                                                                    int(t2//60),
                                                                    int(t2%60),
                                                                    len(n_lst)*len(p_lst)*n_runs))


    t3 = time()-t0

    print('Took {}m {}s to run all {} algorithms'.format(int(t3//60),int(t3%60), algo_type))
    fname = './images/{}_np_{}_pfrac_{}_pquant_{}_ta{}.eps'.format(timestamp,
                                                algo_type, p_frac, p_quant, ta)

    plt.savefig(fname, bbox_inches='tight', pad_inches=0)
    # txt_fname='./results/{}_np_results.txt'.format(timestamp)
    #
    # raw_fname='./results/{}_np_raw.txt'.format(timestamp)
    #
    #
    #
    # if os.path.isfile(raw_fname):
    #     with open(raw_fname, 'a') as f:
    #         # info = '{}_np_{}_pfrac_{}_pquant_{}_ta{}\n'.format(timestamp,
    #         #                                             algo_type, p_frac, p_quant, ta)
    #         # f.write(info)
    #         for i in range(len(algo_lst)):
    #             fps = fp_score[i].flatten()
    #             # print(algo_lst[i])
    #             fp_str = algo_lst[i]+ ', ' + str(ta) + ', ' + outlier_type+', fps, '+''.join(['%0.3f, '])*len(fps)%tuple(fps)
    #             fp_str = fp_str[:-2]+'\n'
    #             f.write(fp_str)
    #             aucs = auc_score[i].flatten()
    #             auc_str = algo_lst[i]+', ' + str(ta) +', ' + outlier_type+', auc, '+ ''.join(['%0.3f, '])*len(aucs)%tuple(aucs)
    #             auc_str = auc_str[:-2]+'\n'
    #             f.write(auc_str)
    # else:
    #     with open(raw_fname, 'w') as f:
    #         info = '{}_np_{}_{}_pfrac_{}_pquant_{}_ta{}\n'.format(timestamp, outlier_type,
    #                                                     algo_type, p_frac, p_quant, ta)
    #         f.write(info)
    #         for i in range(len(algo_lst)):
    #             fps = fp_score[i].flatten()
    #             fp_str = algo_lst[i]+', ' + str(ta) +', ' + outlier_type+', fps, '+''.join(['%0.3f, '])*len(fps)%tuple(fps)
    #             fp_str = fp_str[:-2]+'\n'
    #             f.write(fp_str)
    #             aucs = auc_score[i].flatten()
    #             auc_str = algo_lst[i]+', ' + str(ta) +', ' + outlier_type+', auc, '+ ''.join(['%0.3f, '])*len(aucs)%tuple(aucs)
    #             auc_str = auc_str[:-2]+'\n'
    #             f.write(auc_str)
    #
    # if os.path.isfile(txt_fname):
    #     with open(txt_fname, 'a') as f:
    #         for i in range(len(algo_lst)):
    #             txt = '{}, {}, {}, {}, {}, {}, {},{}, {}, {}, {}, {}, {}\n'.format(timestamp,
    #                                         algo_lst[i], ta, outlier_type, p_frac, p_quant, n_runs,
    #                                         len(n_lst)*len(p_lst)*n_runs, int(10**n_lst[-1]),
    #                                         int(2**p_lst[-1]), int(ts[i]), np.mean(fp_score[i]),
    #                                         np.mean(auc_score[i]))
    #             f.write(txt)
    # else:
    #     with open(txt_fname, 'w') as f:
    #         f.write('timestamp, algo, ta, outlier_type, p_frac, p_quant, n_runs, total_n_runs, max_n, max_p, total_time, fp_score, auc_score\n')
    #         for i in range(len(algo_lst)):
    #             txt = '{}, {}, {}, {}, {}, {},{}, {}, {}, {}, {}, {}, {}\n'.format(timestamp,
    #                                             algo_lst[i], ta, outlier_type, p_frac, p_quant, n_runs,
    #                                             len(n_lst)*len(p_lst)*n_runs, int(10**n_lst[-1]),
    #                                             int(2**p_lst[-1]), int(ts[i]), np.mean(fp_score[i]),
    #                                             np.mean(auc_score[i]))
    #             f.write(txt)
    #
    #

    # plt.show()
    plt.close()

# def record_raw_results(data, filename):
#     if os.path.isfile():


def get_os_data(X, algos):
    """
    X is n,p data set, algos is list of string representations of functions as
    defined in dictionary built below.
    """
    algo_list = [test_IF, test_OP, test_DBSCAN, test_Ridge, test_GRU, test_LSTM, test_OCSVM, test_AE]
    str_algo_lst = ['IF', 'OP', 'DBSCAN', 'Ridge', 'GRU', 'LSTM', 'OC SVM', 'AE']
    fn_dict = {}
    os = []
    for i in len(algo_lst):
        fn_dict[str_algo_lst[i]]=algo_lst[i]
    for algo in algos:
        os.append(algo(X))
    return os

def get_contour_plot(alg_type, algos, tas, score, plot_type):
    """
    uses data alraedy saved to produce plot
    """
    names = ['algo', 'ta', 'score']+list(np.arange(100))
    f_lst = os.listdir('./results/')
    if plot_type == pfq:
        pf_lst = pf_lst
        pq_lst = pq_lst #am referring to hopefully global variable naughty.
    else: #is np plot
        n_lst = n_lst
        p_lst = p_lst


    for f in f_lst:
        if plot_type in f:
            date_time = datetime.datetime.strptime(f[:19], '%Y-%m-%d_%G-%M-%S')
            df = pd.read_csv(f, skipinitialspace=True, index_col=False, header=0) #, names=names)
            n, p = df.shape
            nv = p-3
            names = ['algo', 'ta', 'score']+list(np.arange(nv))
            df.columns = names



def get_occ_data(algo_lst, metric_lst):

    path = os.path.expanduser('~') +'/Data/occupancy/'
    files = ['occ_1', 'occ_2', 'occ_3']
    for file in files:
        df = pd.read_csv(path+file+'.txt')
        print(sum(df['Occupancy']==0)/df.shape[0])
        outs = df.index[df['Occupancy']==1]
        df = df.drop(['date','Occupancy'], axis=1)
        X = df.values
        for algo_s in algo_lst:
            algo = algo_dict[algo_s]
            print('Testing {}'.format(algo_s))
            fp, auc = test_algo(X, outs, algo, metric_lst)
            print('for algo {}, fp = {}, auc = {}'.format(algo_s, fp, auc))

        # print(df.info())

def test_metrics():
    test_case_1 = [1,2,3,4,5,6,7,8]
    auc1 = 1
    fps1 = 0
    outs1 = [7,6,5]
    fpst1 = separation_metric(test_case_1, outs1)
    auct1 = auc(test_case_1, outs1)
    print('fps={}, should be {}, auc={}, should be {}'.format(fpst1, fps1, auct1, auc1))
    test_case_2 = [1,1,1,1,1,1,1,1,1,8]
    auc2 = 1
    fps2 = 0
    outs2 = [9]
    fpst2 = separation_metric(test_case_2, outs2)
    auct2 = auc(test_case_2, outs2)
    print('fps={}, should be {}, auc={}, should be {}'.format(fpst2, fps2, auct2, auc2))
    test_case_3 = [1,1,1,1,1,1,1,1,2,1]
    auc3 = 0.75
    fps3 = 0
    outs3 = [9,8]
    fpst3 = separation_metric(test_case_3, outs3)
    auct3 = auc(test_case_3, outs3)
    print('fps={}, should be {}, auc={}, should be {}'.format(fpst3, fps3, auct3, auc3))
    test_case_4 = [1,2,1,1,1,1,1,1,1,1]
    auc4 = 0.75
    fps4 = 0
    outs4 = [9,1]
    fpst4 = separation_metric(test_case_4, outs4)
    auct4 = auc(test_case_4, outs4)
    print('fps={}, should be {}, auc={}, should be {}'.format(fpst4, fps4, auct4, auc4))

if __name__ == '__main__':

    r = 20
    p_frac = 0.3
    p_quant = 0.3
    # ta = 6
    # n_steps = 10
    n_runs = 10
    gamma = 0.05
    timeout = 900
    noise=.1
    algo_dict = {'VAR':test_VAR, 'FRO':test_OLS, 'FRL':test_LASSO, 'FRR':test_Ridge,
    'GMM': test_GMM, 'OCSVM': test_OCSVM, 'DBSCAN':test_DBSCAN,
    'IF': test_IF,
    'AE': test_AE, 'VAE': test_VAE, 'GRU':test_GRU, 'LSTM':test_LSTM,
    'OP': test_OP, 'GOP': test_GOP, 'SGOP': test_SGOP}
    timestamp = datetime.datetime.fromtimestamp(time())
    timestamp = timestamp.strftime('%Y-%m-%d_%H-%M-%S')
    # timestamp = '2020-11-17_14-33-46' #to continue previously broken expt.

    #quick algos for testing
    reg_algo_lst = ['VAR', 'FRO', 'FRL', 'FRR']
    dens_algo_lst = ['OCSVM', 'GMM', 'DBSCAN', 'IF']
    dl_algo_lst = ['AE', 'VAE', 'LSTM', 'GRU']
    mat_algo_lst = ['OP', 'GOP', 'SGOP']
    algo_type_lst = ['reg', 'dens', 'dl', 'mat']
    lst_lst = [reg_algo_lst, dens_algo_lst, dl_algo_lst, mat_algo_lst]
    # dl_algo_lst = ['LSTM', 'GRU']
    # lst_lst = [reg_algo_lst]
    #to run on occupancy data

    metric_lst = [auc, separation_metric]
    # algo_lst = ['VAR', 'FRO', 'FRL', 'FRR','OCSVM', 'GMM', 'DBSCAN', 'IF', 'AE',
    #             'LSTM', 'GRU', 'OP',  'SGOP']
    # algo_lst= ['VAE']
    # get_occ_data(algo_lst, metric_lst)
    #
    # raise


    # to run on synthetic data.
    p_lst = [1, 2, 3, 4, 5, 6]
    n_lst = [1, 2, 3, 4]
    ta_lst = [1,2,3,4,5,6]
    noise_list=np.arange(0,1.01,0.05)
    # gop_lst = ['OP','GOP']#, 'SGOP']
    # lst_lst = [mat_algo_lst]
    # lst_lst = [['VAR', 'FRR']]
    # lst_lst = [['LSTM', 'GRU']]
    # algo_type_lst=['dl']
    # algo_type_lst = ['reg']
    # p_lst = [1, 2, 3, 4]
    # n_lst = [1, 2, 3]
    #
    # ta_lst = [6]

    #for noise plots.
    pf_lst = np.arange(0.0,1.01,0.2)
    pq_lst = np.arange(0.0,1.01,0.2)
    n = 1000
    p = 32
    metric_lst = [auc, separation_metric]
    metric_str_lst = ['AUC', 'FPs']
    print(timestamp)
    # outlier_type = 'point'
    # for ta in ta_lst:
    #     for i in range(len(lst_lst)):
    #         algo_lst = lst_lst[i]
    #         algo_type = algo_type_lst[i]
    #         plot_noise(p_frac, p_quant, r, gamma, noise_list, ta, n, p, n_runs, algo_lst,algo_type,
    #                          metric_lst, metric_str_lst, timeout, timestamp, outlier_type)


    #
    # for pfq or np plots
    ot_lst = ['point']#, 'context', 'stutter']
    nz_cols = None
    for outlier_type in ot_lst:
        for ta in ta_lst:
            metric_lst = [auc, separation_metric]
            metric_str_lst = ['AUC', 'FPs']
            for i in range(len(lst_lst)): #
                algo_type = algo_type_lst[i]
                algo_lst = lst_lst[i]


                #
                # plot_each_algo_for_np(n_lst, p_lst, r, gamma, noise, ta, p_quant, p_frac,n_runs,
                #                         algo_lst, algo_type, metric_lst, metric_str_lst,
                #                         timeout, timestamp, nz_cols, outlier_type)

                plot_each_algo_for_pfq(pf_lst, pq_lst, r, gamma, noise, ta, n, p, n_runs,
                                            algo_lst, algo_type, metric_lst, metric_str_lst,
                                            timeout, timestamp, outlier_type)
    #
    #                                     # pf_lst, pq_lst, r, gamma, noise, ta, n, p, n_runs,
    #                                     #     algo_list, algo_type, metric, metric_str_lst
    #                                     #     timeout, timestamp, outlier_type
