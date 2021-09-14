#!/usr/bin/env python

import numpy as np
import pandas as pd
from copy import deepcopy
from sklearn.naive_bayes import BernoulliNB


def mcmc(sp_sga_l_v,
         sp_deg_l_v,
         sp_sga_t_v,
         sp_deg_t_v,
         sga_deg_v,
         mcmc=True,
         chal=5,
         fix_mutation=False,
         ite=10000,
         cover_thres=1e-4):
    """Notes for mcmc.

    A iteration which based on sklearn BernoulliNB.

    Args:
        sp_sga_l: 2-D df, index are sample IDs, columns are SGAs or gene names.
        df's value is 0 or 1.
        sp_deg_l: 2-D df, index are sample IDs, columns are DEGs or gene names.
        df's value is 0 or 1.
        sp_sga_t: 2-D df, index are sample IDs, columns are SGAs or gene names.
        df's value is 0 or 1. This one is used for test. Equals to sp_sga_l if
        want to run EM.
        sp_deg_t: 2-D df, index are sample IDs, columns are DEGs or gene names.
        df's value is 0 or 1. This one is used for test. Equals to sp_deg_l if
        want to run EM.
        sga_deg: 2-D df, index are SGAs or gene names, columns are DEGs or gene
        names. df's value is 0 or 1.
        mcmc: Compare the probility matrix with a random matrix, default is True.
        chal: The value used to duplicate df if mcmc is needed. default is 5.
        fix_mutation: Wether fixed mutaion in sp_sga. default is True.
        ite: Max iteration times. default is 10000.
        cover_thres: The value which used to see the coverigence. default is 1e-4.


    Returns:
        sp_pro_t: 2-D df, index are sample IDs, columns are SGAs or gene names.
        df's value is 0 or 1. this df is the hidden variable's or protein's
        prediction.
        pa1_ll: Two layer list which are used to save p(SGA=1), The first
        layer orders from sp_sga's columns, the second layer orders from
        iteration times.
    """
    # Avoid changing input
    sp_sga_l = deepcopy(sp_sga_l_v)
    sp_sga_t = deepcopy(sp_sga_t_v)
    sp_deg_l = deepcopy(sp_deg_l_v)
    sp_deg_t = deepcopy(sp_deg_t_v)
    sga_deg = deepcopy(sga_deg_v)

    # decrease data size
    sga_deg = sga_deg.loc[:, (np.sum(sga_deg, 0) >=
                              1)]  # Decrease the size of sga_deg.
    sp_deg_l = sp_deg_l[sga_deg.columns]
    sp_deg_t = sp_deg_t[sga_deg.columns]

    # add pre-add
    if 'sga:' not in sp_sga_l.columns[0]:
        sp_sga_l.columns = ['sga:' + ele for ele in sp_sga_l.columns]
    if 'sga:' not in sp_sga_t.columns[0]:
        sp_sga_t.columns = ['sga:' + ele for ele in sp_sga_t.columns]
    if 'deg:' not in sp_deg_l.columns[0]:
        sp_deg_l.columns = ['deg:' + ele for ele in sp_deg_l.columns]
    if 'deg:' not in sp_deg_t.columns[0]:
        sp_deg_t.columns = ['deg:' + ele for ele in sp_deg_t.columns]
    if 'sga:' not in sga_deg.index[0]:
        sga_deg.index = ['sga:' + ele for ele in sga_deg.index]
    if 'deg:' not in sga_deg.columns[0]:
        sga_deg.columns = ['deg:' + ele for ele in sga_deg.columns]

    pa1_ll = []
    sp_pro_t = np.zeros((sp_sga_t.shape[0], 0))

    for sga in sp_sga_l.columns:

        pa1_l = []
        deg_l = [ele for ele in sga_deg.columns if sga_deg.loc[sga, ele] == 1]
        sga_l = list((np.sum(sga_deg[deg_l], 1) > 0).index).remove(sga)

        sp_sga_dl = np.tile(sp_sga_l.loc[:, [sga]].values, (chal, 1))
        # np.tile(df, (chal, 1)) is dulpicate size
        sp_dsga_dl = np.tile(sp_sga_l.loc[:, [sga_l]].values, (chal, 1))
        sp_deg_dl = np.tile(sp_deg_l.loc[:, deg_l].values, (chal, 1))
        sp_deg_dl = np.hstack((sp_dsga_dl, sp_deg_dl))
        sp_sga_dt = np.tile(sp_sga_t.loc[:, [sga]].values, (chal, 1))
        sp_dsga_dt = np.tile(sp_sga_t.loc[:, [sga_l]].values, (chal, 1))
        sp_deg_dt = np.tile(sp_deg_t.loc[:, deg_l].values, (chal, 1))
        sp_deg_dt = np.hstack((sp_dsga_dt, sp_deg_dt))

        NB = BernoulliNB().fit(sp_deg_dl, np.ravel(sp_sga_dl))

        for i in range(ite):

            sp_pro_dt = NB.predict_proba(sp_deg_dt)[:, [1]].reshape(
                [sp_sga_dt.shape[0], 1])
            # NB.predict_proba() give two columns probability.

            if mcmc is True:
                mc_m = np.random.rand(sp_pro_dt.shape[0], sp_pro_dt.shape[1])
                sp_pro_dt[(sp_pro_dt) >= mc_m] = 1
                sp_pro_dt[(sp_pro_dt) < mc_m] = 0
            else:
                sp_pro_dt = sp_pro_dt

            if fix_mutation is True:
                sp_pro_dt[(sp_pro_dt + sp_sga_dt) >= 1] = 1
                sp_pro_dt[(sp_pro_dt + sp_sga_dt) < 1] = 0
            else:
                sp_pro_dt = sp_pro_dt

            pa1_l.append(np.mean(sp_pro_dt))

            if i > 0:
                if abs(pa1_l[i] - pa1_l[i - 1]) < cover_thres:
                    break

            NB = BernoulliNB().fit(sp_deg_dt, np.ravel(sp_pro_dt))
            # np.ravel() covert 2-D array into list

        pa1_ll.append(pa1_l)
        sp_pro_t = np.hstack(
            (sp_pro_t,
             np.mean(sp_pro_dt.reshape([chal, sp_pro_t.shape[0], 1]), 0)))

    sp_pro_t = pd.DataFrame(sp_pro_t,
                            index=sp_sga_t.index,
                            columns=sp_sga_t.columns)

    return sp_pro_t, pa1_ll
