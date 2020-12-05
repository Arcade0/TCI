#!/usr/bin/env python

import numpy as np
import pandas as pd
from copy import deepcopy
from sklearn.model_selection import KFold
from sklearn.naive_bayes import  GaussianNB, BernoulliNB,MultinomialNB
from sklearn.decomposition import PCA
from multiprocessing import Process, Pool, cpu_count
from scipy.stats import multivariate_normal

def em(
    sp_sga_l_v, sp_deg_l_v, sp_sga_t_v, sp_deg_t_v, sga_deg_v, 
    mcmc=True, chal=5, fix_mutation=False, ite=10000, cover_thres=1, clf="b"):
    
    """Notes for EM.

    A iteration which based on sklearn NB.

    Args:
        sp_sga_l: 2-D df, index are sample IDs, columns are SGAs or gene names. df's value is 0 or 1.
        sp_deg_l: 2-D df, index are sample IDs, columns are DEGs or gene names. df's value is 0 or 1.
        sp_sga_t: 2-D df, index are sample IDs, columns are SGAs or gene names. df's value is 0 or 1. This one 
                  is used for test. Equals to sp_sga_l if want to run EM.
        sp_deg_t: 2-D df, index are sample IDs, columns are DEGs or gene names. df's value is 0 or 1. This one 
                  is used for test. Equals to sp_deg_l if want to run EM.
        sga_deg: 2-D df, index are SGAs or gene names, columns are DEGs or gene names. df's value is 0 or 1.
        mcmc: Compare the probility matrix with a random matrix, default is True.
        chal: The value used to duplicate df if mcmc is needed. default is 5.
        fix_mutation: Wether fixed mutaion in sp_sga. default is True. 
        ite: Max iteration times. default is 10000.
        cover_thres: The value which used to see the coverigence. default is 1e-4.
        clf: model used in NB, "g" = GaussianNB(), "b" = BernoulliNB(), "m" = MultinomialNB().
        

    Returns:
        sp_pro_t: 2-D df, index are sample IDs, columns are SGAs or gene names. df's value is 0 or 1.
                  this df is the hidden variable's or protein's prediction.
        pa1_ll: Two layer list which are used to save p(SGA=1), The first layer orders from sp_sga's columns, 
                the second layer orders from iteration times.
    """
    
    # Avoid changing input
    sp_sga_l = deepcopy(sp_sga_l_v)
    sp_sga_t = deepcopy(sp_sga_t_v)
    sp_deg_l = deepcopy(sp_deg_l_v)
    sp_deg_t = deepcopy(sp_deg_t_v)
    sga_deg = deepcopy(sga_deg_v)
    print("load successful")
    e=1e-5  # Avoid overflowing in log.
    sga_deg = sga_deg.loc[:, (np.sum(sga_deg, 0) >= 1)]  # Decrease the size of sga_deg.
    sp_deg_l = sp_deg_l[sga_deg.columns]
    sp_deg_t = sp_deg_t[sga_deg.columns]

    pa1_ll = []
    sp_pro_t = np.zeros((sp_sga_t.shape[0], 0))

    clf_d = {"g":GaussianNB(), "b":BernoulliNB(), "m":MultinomialNB()}
    clf = clf_d[clf]

    for sga in sp_sga_l.columns:
#         print(sga)

        pa1_l = []
        deg_l = [ele for ele in sga_deg.columns if sga_deg.loc[sga, ele] == 1]
        sp_sga_dl = np.tile(sp_sga_l.loc[:, [sga]].values, (chal, 1)) # np.tile(df, (chal, 1)) is dulpicate size 
        sp_deg_dl = np.tile(sp_deg_l.loc[:, deg_l].values, (chal, 1))
        sp_sga_dt = np.tile(sp_sga_t.loc[:, [sga]].values, (chal, 1))
        sp_deg_dt = np.tile(sp_deg_t.loc[:, deg_l].values, (chal, 1))
        
        NB = clf.fit(sp_deg_dl, np.ravel(sp_sga_dl))
        for i in range(ite):
            
            sp_pro_dt = NB.predict(sp_deg_dt).reshape([sp_sga_dt.shape[0], 1])  # NB.predict_proba() give two columns probability.

            if mcmc == True:
                mc_m = np.random.rand(sp_pro_dt.shape[0], sp_pro_dt.shape[1])
                sp_pro_dt[(sp_pro_dt) >= mc_m] = 1
                sp_pro_dt[(sp_pro_dt) < mc_m] = 0
            else:
                sp_pro_dt = sp_pro_dt

            if fix_mutation == True:
                sp_pro_dt[(sp_pro_dt + sp_sga_dt) >= 1] = 1
                sp_pro_dt[(sp_pro_dt + sp_sga_dt) < 1] = 0
            else:
                sp_pro_dt = sp_pro_dt
            
            e = 1e-5
            ll = np.log(np.dot(sp_pro_dt.T, sp_deg_t)/sp_pro_dt.shape[0] + e)*np.dot(sp_pro_dt.T, sp_deg_t) + \
                np.log(np.dot(sp_pro_dt.T, 1-sp_deg_t)/sp_pro_dt.shape[0] + e)*np.dot(sp_pro_dt.T, 1-sp_deg_t) + \
                np.log(np.dot(1-sp_pro_dt.T, sp_deg_t)/sp_pro_dt.shape[0] + e)*np.dot(1-sp_pro_dt.T, sp_deg_t) + \
                np.log(np.dot(1-sp_pro_dt.T, 1-sp_deg_t)/sp_pro_dt.shape[0] + e)*np.dot(1-sp_pro_dt.T, 1-sp_deg_t)

            pa1_l.append(np.mean(ll))

            if i > 0:
#                 print(abs(pa1_l[i] -pa1_l[i - 1]))
                
                if abs(pa1_l[i] -pa1_l[i - 1]) < cover_thres:
                    break

            NB = clf.fit(sp_deg_dt, np.ravel(sp_pro_dt))  # np.ravel() covert 2-D array into list
           
            
        pa1_ll.append(pa1_l)
        sp_pro_t = np.hstack(
            (sp_pro_t, np.mean(
                sp_pro_dt.reshape([chal, sp_pro_t.shape[0], 1]), 0)))

    sp_pro_t = pd.DataFrame(sp_pro_t, index=sp_sga_t.index, columns=sp_sga_t.columns)
    
    return sp_pro_t, pa1_ll

def em_fold(
    sp_sga_v, sp_deg_v, sga_deg_v, mcmc=True, chal=5, 
    fix_mutation=False, ite=10000, cover_thres=1, clf="b", fold=10):

    """Notes for EM fold.

    A iteration which based on sklearn BernoulliNB.

    Args:
        sp_sga: 2-D df, index are sample IDs, columns are SGAs or gene names. df's value is 0 or 1.
        sp_deg: 2-D df, index are sample IDs, columns are DEGs or gene names. df's value is 0 or 1.
        sga_deg: 2-D df, index are SGAs or gene names, columns are DEGs or gene names. df's value is 0 or 1.
        mcmc: Compare the probility matrix with a random matrix, default is True.
        fix_mutation: Wether fixed mutaion in sp_sga. default is True. 
        ite: Max iteration times. default is 10000.
        cover_thres: The value which used to see the coverigence. default is 1e-4.
        chal: The value used to duplicate df if mcmc is needed. default is 5.
        fold: Numers used to test, defalut is 10.

    Returns:
        sp_pro_p: 2-D df, index are sample IDs, columns are SGAs or gene names. df's value is 0 or 1.
                  this df is the hidden variable's or protein's prediction.
    """

    # Avoid changing input
    sp_sga = deepcopy(sp_sga_v)
    sp_deg = deepcopy(sp_deg_v)
    sga_deg = deepcopy(sga_deg_v)

    kf = KFold(n_splits=fold)
    kf.get_n_splits(sp_deg)

    sp_pro = pd.DataFrame(np.zeros((0, sp_sga.shape[1])), columns=sp_sga.columns)

    for train_index, test_index in kf.split(sp_sga):

        sp_sga_l, sp_deg_l = sp_sga.iloc[train_index, :], sp_deg.iloc[train_index, :]
        sp_sga_t, sp_deg_t = sp_sga.iloc[test_index, :], sp_deg.iloc[test_index, :]

        sp_pro_l, pa1_ll = em(
            sp_sga_l, sp_deg_l, sp_sga_l, sp_deg_l, sga_deg, 
            mcmc=mcmc, chal=chal, fix_mutation=fix_mutation, ite=ite, cover_thres=cover_thres, clf=clf)
        sp_pro_l = sp_pro_l > 0.5

        sp_pro_t, pa1_ll = em(
            sp_pro_l, sp_deg_l, sp_sga_t, sp_deg_t, sga_deg, 
            mcmc=mcmc, chal=chal, fix_mutation=fix_mutation, ite=1, cover_thres=cover_thres, clf=clf)
        sp_pro = pd.concat([sp_pro, sp_pro_t], 0)

    return sp_pro

def gmm(
    sp_sga_l_v, sp_deg_l_v, sp_sga_t_v, sp_deg_t_v, sga_deg_v, 
    mcmc=True, chal=5, fix_mutation=False,  ite=10000, cover_thres=1e-4):

    """Notes for GMM.

    A iteration which based on sklearn BernoulliNB.

    Args:
        sp_sga_l: 2-D df, index are sample IDs, columns are SGAs or gene names. df's value is 0 or 1.
        sp_deg_l: 2-D df, index are sample IDs, columns are DEGs or gene names. df's value is 0 or 1.
        sp_sga_t: 2-D df, index are sample IDs, columns are SGAs or gene names. df's value is 0 or 1. This one 
            is used for test. Equals to sp_sga_l if want to run EM.
        sp_deg_t: 2-D df, index are sample IDs, columns are DEGs or gene names. df's value is 0 or 1. This one 
            is used for test. Equals to sp_deg_l if want to run EM.
        sga_deg: 2-D df, index are SGAs or gene names, columns are DEGs or gene names. df's value is 0 or 1.
        mcmc: Compare the probility matrix with a random matrix, default is True.
        fix_mutation: Wether fixed mutaion in sp_sga. default is True. 
        ite: Max iteration times. default is 10000.
        cover_thres: The value which used to see the coverigence. default is 1e-4.
        chal: The value used to duplicate df if mcmc is needed. default is 5.

    Returns:
        sp_pro_t: 2-D df, index are sample IDs, columns are SGAs or gene names. df's value is 0 or 1.
                  this df is the hidden variable's or protein's prediction.
        pa1_ll: Two layer list which are used to save p(SGA=1), 
                The first layer orders from sp_sga's columns, the second layer orders from iteration times.
    """

    # Avoid changing input
    sp_sga_l = deepcopy(sp_sga_l_v)
    sp_sga_t = deepcopy(sp_sga_t_v)
    sp_deg_t = deepcopy(sp_deg_l_v)
    sp_deg_t = deepcopy(sp_deg_t_v)
    sga_deg = deepcopy(sga_deg_v)

    e=1e-5  # Avoid overflowing in log.
    sga_deg = sga_deg.loc[:, (np.sum(sga_deg, 0) >= 1)]  # Decrease the size of sga_deg.
    sp_deg_l = sp_deg_l[sga_deg.columns]
    sp_deg_t = sp_deg_t[sga_deg.columns]

    pa1_ll = []
    sp_pro_t = np.zeros((sp_sga_t.shape[0], 0))
    pca = PCA(n_components="mle")

    for sga in sp_sga_l.columns:

        pa1_l = []
        deg_l = [ele for ele in sga_deg.columns if sga_deg.loc[sga, ele] == 1]
        
        sp_sga_dl = np.tile(sp_sga_l.loc[:, [sga]].values, (chal, 1)) # np.tile(df, (chal, 1)) is dulpicate size 
        sp_deg_dl = np.tile(sp_deg_l.loc[:, deg_l].values, (chal, 1))
        pca.fit(sp_deg_dl)
        sp_deg_dl = pca.transform(sp_deg_dl)

        sp_sga_dt = np.tile(sp_sga_t.loc[:, [sga]].values, (chal, 1))
        sp_deg_dt = np.tile(sp_deg_t.loc[:, deg_l].values, (chal, 1))
        pca.fit(sp_deg_dt)
        sp_deg_dt = pca.transform(sp_deg_dt)


        # Initial value
        pa1 = np.sum(sp_sga_dl, 0) / sp_sga_dl.shape[0]
        u1 = np.sum(sp_sga_dl * sp_deg_dl, 0) / np.sum(sp_sga_dl, 0)
        gama1 = np.dot(
            (sp_sga_dl * (sp_deg_dl - u1)).T, 
            sp_sga_dl * (sp_deg_dl - u1)) / np.sum(sp_sga_dl, 0)

        pa0 = 1 - pa1
        u0 = np.sum((1 - sp_sga_dl) * sp_deg_dl, 0) / np.sum((1 - sp_sga_dl), 0)
        gama0 = np.dot(
            ((1 - sp_sga_dl) * (sp_deg_dl - u0)).T, 
            (1 - sp_sga_dl) * (sp_deg_dl - u0)) / np.sum((1 - sp_sga_dl), 0)


        for i in range(ite):

            var1 = multivariate_normal(mean=list(u1), cov=gama1)
            var0 = multivariate_normal(mean=list(u0), cov=gama0)

            # E step, set Qt = p(A|D) to get the likelihood in this set.
            logpa1 = np.log(pa1 + e)
            logpa0 = np.log(pa0 + e)
            logpa1d = np.log(var1.pdf(sp_deg_dt) + e)[:, None]
            logpa0d = np.log(var0.pdf(sp_deg_dt) + e)[:, None]
            sp_pro_dt = (1 / (1 + np.exp(logpa0d - logpa1d)))

            # M step, find new theta to maximum lower bound.
            pa1 = np.sum(sp_pro_dt, 0) / sp_pro_dt.shape[0]
            u1 = np.sum(sp_pro_dt * sp_deg_dt, 0) / np.sum(sp_pro_dt, 0)
            gama1 = np.dot((sp_pro_dt * (sp_deg_dt - u1)).T, 
                            sp_pro_dt * (sp_deg_dt - u1)) / np.sum(sp_pro_dt, 0)

            pa0 = 1 - pa1
            u0 = np.sum((1 - sp_pro_dt) * sp_deg_dt, 0) / np.sum((1 - sp_pro_dt), 0)
            gama0 = np.dot(((1 - sp_pro_dt) * (sp_deg_dt - u0)).T, 
                            (1 - sp_pro_dt) * (sp_deg_dt - u0)) / np.sum((1 - sp_pro_dt), 0)

            if mcmc == True:
                mc_m = np.random.rand(sp_pro_dt.shape[0], sp_pro_dt.shape[1])
                sp_pro_dt[(sp_pro_dt) >= mc_m] = 1
                sp_pro_dt[(sp_pro_dt) < mc_m] = 0
            else:
                sp_pro_dt = sp_pro_dt

            if fix_mutation == True:
                sp_pro_dt[(sp_pro_dt + sp_sga_dt) >= 1] = 1
                sp_pro_dt[(sp_pro_dt + sp_sga_dt) < 1] = 0
            else:
                sp_pro_dt = sp_pro_dt

            pa1_l.append(np.mean(sp_pro_dt))

            if i > 0:
                if abs(pa1_l[i] - pa1_l[i - 1]) < cover_thres:
                    break

        pa1_ll.append(pa1_l)
        sp_pro_t = np.hstack(
            (sp_pro_t, np.mean(
                sp_pro_dt.reshape([chal, sp_sga_t.shape[0], 1]), 0)))

    sp_pro_t = pd.DataFrame(sp_pro_t, index=sp_sga_t.index, columns=sp_sga_t.columns)
    
    return sp_pro_t, pa1_ll

