import pandas as pd
import numpy as np
import sys
import os
from copy import deepcopy
import time


def EM(S_Am, S_Dm, A_D, fix_mutation=1, e=1e-5, cut_value=0.85, ite=10000, cover_thres=1e-4):
    
    S_Ad = S_Am.values
    S_Dd = S_Dm.values
    A_Dd = A_D.values
    logpD1A1_l = [1]
    
    for i in range(ite):
        
        pA1 = (np.sum(S_Ad, 0)[:, None] + 1) / (S_Ad.shape[0] + 2)
        pD1A1 = (np.dot(S_Ad.T, S_Dd) + 1) / (np.sum(S_Ad, 0)[:, None] + 2)
        pD1A0 = (np.dot(1 - S_Ad.T, S_Dd) + 1) / (np.sum(1 - S_Ad, 0)[:, None] + 2)
        
        logpA1 = np.log(pA1 + e)
        logpA0 = np.log(1 - pA1 + e)
        logpD1A1 = np.log(pD1A1 + e) * A_Dd
        logpD0A1 = np.log(1 - pD1A1 + e) * A_Dd
        logpD1A0 = np.log(pD1A0 + e) * A_Dd
        logpD0A0 = np.log(1 - pD1A0 + e) * A_Dd
        
        logpA1D = np.dot(S_Dd, logpD1A1.T) + \
                  np.dot(1 - S_Dd, logpD0A1.T) + logpA1.T
        logpA0D = np.dot(S_Dd, logpD1A0.T) + \
                  np.dot(1 - S_Dd, logpD0A0.T) + logpA0.T
        S_Ad = 1 / (1 + np.exp(logpA0D - logpA1D))
        
        if fix_mutation == 1:
            S_Ad = S_Ad + S_Am
            S_Ad[S_Ad > 1] = 1
        else:
            S_Ad = S_Ad
        
        logpD1A1_l.append(logpD1A1)
        print(len(logpD1A1_l))
        print(np.sum((logpD1A1_l[i]-logpD1A1_l[i-1])**2))
        if np.sum((logpD1A1_l[i] - logpD1A1_l[i - 1]) ** 2) < cover_thres:
            break
    
    S_Ad0 = pd.DataFrame(S_Ad, index=S_Am.index, columns=S_Am.columns)
    S_Ad1 = deepcopy(S_Ad0)
    S_Ad1[S_Ad1 > cut_value] = 1
    S_Ad1[S_Ad1 <= cut_value] = 0
    
    prm = [logpA1, logpA0, logpD1A1, logpD0A1, logpD1A0, logpD0A0]
    
    return S_Ad0, S_Ad1, prm

def prei(S_Ad,S_Dm, A_D, A_Dc, S_Dc):
    spA1 = pd.DataFrame(
        (np.sum(S_Ad, 0)[:, None] + 1) / (S_Ad.shape[0] + 2), index=A_D.index).loc[A_Dc.index, :]
    spD1A1 = pd.DataFrame((np.dot(S_Ad.T, S_Dm) + 1) / (np.sum(S_Ad, 0)[
                                                        :, None] + 2), index=A_D.index, columns=A_D.columns).loc[
        A_Dc.index, A_Dc.columns]
    spD1A0 = pd.DataFrame((np.dot(1 - S_Ad.T, S_Dm) + 1) / (np.sum(1 - S_Ad, 0)[
                                                            :, None] + 2), index=A_D.index, columns=A_D.columns).loc[
        A_Dc.index, A_Dc.columns]
    sAclist = []
    for i in range(spA1.shape[0]):
        spA1i = spA1.iloc[i, :]
        spD1A1i = spD1A1.loc[spA1.index[i], A_Dlist[i].columns]
        spD1A0i = spD1A0.loc[spA1.index[i], A_Dlist[i].columns]
        S_Dci = S_Dc.loc[:, A_Dlist[i].columns]
        e = 1e-5
        slogpA1Di = np.dot(S_Dci, np.log(spD1A1i + e).T) + np.dot(1 -
                                                                  S_Dci, np.log(1 - spD1A1i + e).T) + np.array(
            np.log(spA1i)).T
        slogpA0Di = np.dot(S_Dci, np.log(spD1A0i + e).T) + np.dot(1 -
                                                                  S_Dci, np.log(1 - spD1A0i + e).T) + np.array(
            np.log(1 - spA1i)).T
        S_Adci = 1 / (1 + np.exp(slogpA0Di - slogpA1Di))
        sAclist.append(S_Adci)
    S_Adc0 = pd.DataFrame(sAclist, index=S_Ac.columns, columns=S_Ac.index).T
    S_Adc1 = (S_Adc0 > 0.85)
    S_Adc2 = S_Adc1 + S_Ac
    S_Adc2[S_Adc2 > 1] = 1
    return S_Adc2

def dr_sen(func, fix_mutation=1, e=1e-5, cut_value=0.85, ite=10000, cover_thres=1e-4):
    S_Ac = pd.read_csv('DrugNormal/Output/nnGDSCmut.csv', header=0, index_col=0)
    A_D = pd.read_csv('EMva/Input/A_D.csv', header=0, index_col=0)
    S_Dc = pd.read_csv('DrugNormal/Output/nGDSCexp.csv', header=0, index_col=0)
    A_Dc = A_D.loc[S_Ac.columns, A_D.columns & S_Dc.columns]
    S_Dc = S_Dc.loc[:, A_Dc.columns]
    if func == 'em':
        S_Am=pd.read_csv('EMva/Input/S_Am.csv', header=0, index_col=0)
        S_Dm = pd.read_csv('EMva/Input/S_Dm.csv', header=0, index_col=0)
        A_D = pd.read_csv('EMva/Input/A_D.csv', header=0, index_col=0)
        S_Ad0, S_Ad1, prm = EM(S_Am, S_Dm, A_D, fix_mutation, e,cut_value, ite, cover_thres)
        S_Ad1.to_csv('EMva/Output/S_Ad1.csv', header=True, index=True)
    
    # use TCGA DEG run TCGA EM
    if func == 'ttg':
        S_Ad = pd.read_csv('EMva/Output/S_Ad1.csv', header=0, index_col=0)
        S_Dm = pd.read_csv('EMva/Input/S_Dm.csv', header=0, index_col=0)
        S_Attg = prei(S_Ad, S_Dm, A_D, A_Dc, S_Dc)
        S_Attg.to_csv('EMva/Output/S_Attg.csv', header=True, index=True)
        
    # use GDSC DEG run TCGA EM
    if func == 'tgg':
        S_Am = pd.read_csv('EMva/Input/S_Am.csv', header=0, index_col=0)
        S_Dm = pd.read_csv('EMva/Input/S_Dm.csv', header=0, index_col=0)
        S_Atgg0, S_Atgg1, prm = EM(S_Am.loc[:, A_Dc.index], S_Dm.loc[:,
                                                            A_Dc.columns], A_D.loc[A_Dc.index, :], fix_mutation, e,
                                   cut_value, ite, cover_thres)
        S_Atgg1 = prei(S_Atgg1, A_Dc, S_Dc)
        S_Atgg1.to_csv('EMva/Output/S_Atgg1.csv', header=True, index=True)
        
    # use GDSC run EM
    if func == 'ggg':
        S_Aggg0, S_Aggg1, prm = EM(S_Ac, S_Dc, A_Dc, fix_mutation, e, cut_value, ite, cover_thres)
        S_Aggg1.to_csv('EMva/Output/S_Aggg1.csv', header=True, index=True)
    
    # use GDSC run fges
    if func == 'ggg_fges':
        S_Aggg0, S_Aggg1, prm = EM(
            S_Ac, S_Dc, A_Dc, fix_mutation, e, cut_value, ite, cover_thres)
        
        A_Dc.index = ['sga:' + idx for idx in A_Dc.index]
        A_Dc.columns = ['dge:' + col for col in A_Dc.columns]
        S_Ac.columns = A_Dc.index
        S_Dc.columns = A_Dc.columns
        S_Aggg1.columns = A_Dc.index
        S_Aggg1.to_csv('EMva/Input/S_Aggg1.csv', header=True, index=True)
        S_Ac.to_csv('EMva/Input/S_Amggg.csv', header=True, index=True)
        S_Dc.to_csv('EMva/Input/S_Dmggg.csv', header=True, index=True)
        A_Dc.to_csv('EMva/Input/A_Dggg.csv', header=True, index=True)
        
        os.system(
            'python fges_stem.py EMva/Input/S_Amggg.csv EMva/Input/S_Dmggg.csv EMva/Input/A_Dggg.csv noEM EMva/Output/S_Aggg1.csv PI3K Pathway/GDSC 30')
        
    # use GDSC run 10 EM
    if func == 'ggg10':
        S_Agggt0, S_Agggt1 = EM10fold(S_Ac, S_Dc, A_Dc)
        S_Agggt1.to_csv('EMva/Output/S_Agggt1.csv', header=True, index=True)
        
    # random A_D
    if func == 'ggg_rad':
        for i in range(7,10):
            A_D = pd.read_csv('EMva/Input/A_D_rad'+'-'+str(i)+'.csv', header=0, index_col=0)
            A_Dc = A_D.loc[S_Ac.columns, A_D.columns & S_Dc.columns]
            S_Dc = S_Dc.loc[:, A_Dc.columns]
        
            S_Ad0_rad, S_Ad1_rad, prm = EM(S_Ac, S_Dc, A_Dc, fix_mutation, e, cut_value, ite, cover_thres)
            S_Ad1_rad.to_csv('EMva/Output/S_Ad1_rad'+'-'+str(i)+'.csv', header=True, index=True)
    
    # random A_D fges
    if func == 'ggg_rd_fges':
        A_D = pd.read_csv('EMva/Input/A_D_rd.csv', header=0, index_col=0)
        A_Dc = A_D.loc[S_Ac.columns, A_D.columns & S_Dc.columns]
        S_Dc = S_Dc.loc[:, A_Dc.columns]
        
        S_Ad0_rd, S_Ad1_rad, prm = EM(S_Ac, S_Dc, A_Dc, fix_mutation, e, cut_value, ite, cover_thres)
        S_Ad1_rad.to_csv('EMva/Output/S_Ad1_rad.csv', header=True, index=True)
        
        A_Dc.index = ['sga:' + idx for idx in A_Dc.index]
        A_Dc.columns = ['dge:' + col for col in A_Dc.columns]
        
        S_Ac.columns = A_Dc.index
        S_Dc.columns = A_Dc.columns
        S_Ad1_rad.columns = A_Dc.index
        S_Ad1_rad.to_csv('EMva/Input/S_Ad1_rad.csv', header=True, index=True)
        S_Ac.to_csv('EMva/Input/S_Am_rd.csv', header=True, index=True)
        S_Dc.to_csv('EMva/Input/S_Dm_rd.csv', header=True, index=True)
        A_Dc.to_csv('EMva/Input/A_D_rd.csv', header=True, index=True)
        
        os.system(
            'python fges_stem.py EMva/Input/S_Am_rd.csv EMva/Input/S_Dm_rd.csv EMva/Input/A_D_rd.csv noEM EMva/Output/S_Ad1_rad.csv PI3K Pathway/GDSC_rd 30')
        
    # random S_A
    if func == 'ggg_ra':
        S_Ac = pd.read_csv('EMva/Input/nGDSCmut_rd.csv',
                           header=0, index_col=0).T
        print(np.sum(S_Ac, 0))
        
        S_Ad0_ra, S_Ad1_ra, prm = EM(
            S_Ac, S_Dc, A_Dc, fix_mutation, e, cut_value, ite, cover_thres)
        S_Ad1_ra.to_csv('EMva/Output/S_Ad1_ra.csv', header=True, index=True)
        
    # all DEG
    if func == 'ggg_ad':
        A_D = pd.DataFrame(np.ones((A_D.shape[0], A_D.shape[1])),
                           index=A_D.index, columns=A_D.columns)
        A_Dc = A_D.loc[S_Ac.columns, A_D.columns & S_Dc.columns]
        S_Dc = S_Dc.loc[:, A_Dc.columns]
        
        S_Ad0_ad, S_Ad1_ad, prm = EM(
            S_Ac, S_Dc, A_Dc, fix_mutation, e, cut_value, ite, cover_thres)
        S_Ad1_ad.to_csv('EMva/Output/S_Ad1_ad.csv', header=True, index=True)
        
    # partial DEG
    if func == 'ggg_se':
        A_D = pd.read_csv('EMva/Input/A_D_se.csv', header=0, index_col=0)
        A_Dc = A_D.loc[S_Ac.columns, A_D.columns & S_Dc.columns]
        S_Dc = S_Dc.loc[:, A_Dc.columns]
        
        S_Ad0_se, S_Ad1_se, prm = EM(
            S_Ac, S_Dc, A_Dc, fix_mutation, e, cut_value, ite, cover_thres)
        S_Ad1_se.to_csv('EMva/Output/S_Ad1_se.csv', header=True, index=True)
        
    # all DEG
    if func == 'ggg_ae':
        A_D = pd.read_csv('EMva/Input/A_D_ae.csv', header=0, index_col=0)
        A_Dc = A_D.loc[S_Ac.columns, A_D.columns & S_Dc.columns]
        S_Dc = S_Dc.loc[:, A_Dc.columns]
        
        S_Ad0_sr, S_Ad1_sr, prm = EM(
            S_Ac, S_Dc, A_Dc, fix_mutation, e, cut_value, ite, cover_thres)
        S_Ad1_sr.to_csv('EMva/Output/S_Ad1_ae.csv', header=True, index=True)

# sta
def sta(input_name, output_name, tumor_type, cut):
    
    # input_name
    if 'nGDSCmut' in input_name:
        pred = pd.read_csv(input_name, header=0, index_col=0)
        pred.index = [int(idx) for idx in pred.index]
        
    if 'All' in input_name:
        mut = pd.read_csv('DrugNormal/Output/nGDSCmut.csv', header=0, index_col=0)
        mut.index = [int(idx) for idx in mut.index]
        pred = pd.DataFrame(np.ones((mut.shape[0], mut.shape[1])), index=mut.index, columns=mut.columns)
        
    if 'Pathway' in input_name:
        mut = pd.read_csv('DrugNormal/Output/nGDSCmut.csv', header=0, index_col=0)
        pred = pd.read_csv(input_name, header=0, index_col=None)
        pred.columns = [ele[4:] for ele in pred.columns]
        mut.index = [int(idx) for idx in mut.index]
        pred.index = mut.index
        new_col = [col for col in mut.columns if col in pred.columns]
        pred = pred[new_col]
        
    if 'S_A' in input_name:
        pred = pd.read_csv(input_name, header=0, index_col=0)
    label = pd.read_csv('DrugNormal/Input/GDSC drug response.csv', header=0, index_col=0)
    
    # tumor_type
    if tumor_type == 1:
        ttcn = pd.read_csv('DrugNormal/Input/ttcn.csv', header=0, index_col=0)
        tt_row = [ele for ele in pred.index if ele in ttcn.iloc[:, 0].values]
        pred = pred.loc[tt_row]
        label = label.loc[tt_row]
    
    
    drug_l = ['171','326','228','1053','86',
              '1502','1030','152',
              '1061','1036','1371','1373','29','159','30',
              '292','293','5','298',
              '54','1054','219',
              '1','1010','1114','282','119','1032','1377','273','1143',
              '255','1049','155','308',
              '269','269-1047','1133-1047',
              '134',
              '1067',
              '223-1066-224-238','1058-1066-224-238','1527-1066-224-238','302-1066-224-238','283-1066-224-238','1057-1066-224-238',
              '3','83','299','1016','1059','1166']
    gene_l = ['AKT1','AKT1','AKT1','AKT1','AKT1',
              'AR','ATM','ATM',
              'BRAF','BRAF+RAF1','BRAF+RAF1','BRAF+RAF1','BRAF+RAF1','BRAF','BRAF+RAF1+PDGFRA+KIT',
              'KIT+PDGFRA','KIT+PDGFRA','KIT+PDGFRA','KIT+KDR',
              'CDK4','CDK4','CDK4',
              'EGFR','EGFR','EGFR','EGFR','EGFR+ERBB2','EGFR+ERBB2','EGFR+ERBB2','EGFR+ERBB2','EGFR',
              'ERBB2','FGFR1+FGFR3','FGFR1+FGFR3+PDGFRA','FGFR1+FGFR3+PDGFRA+KDR',
              'MDM4','MDM4','TP53',
              'TOP2A',
              'PPM1D',
              'PIK3CA','PIK3CA','PIK3CA','PIK3CA','PIK3CA+MTOR','PIK3CA+MTOR',
              'MTOR','MTOR','MTOR','MTOR','MTOR','MTOR']
    
    bdf = np.zeros((label.shape[0],0))
    for drug in drug_l:
        if '-' in drug:
            drug_sl = drug.split('-')
            for i in range(len(drug_sl)):
                if i == 0:
                    df = label[[drug_sl[i]]].values
                else:
                    df = df - label[[drug_sl[i]]].values
            bdf = np.concatenate((bdf, df),axis=1)
        else:
            df = label[[drug]].values
            bdf = np.concatenate((bdf, df),axis=1)
    bdf[bdf<0] = 0
    label_col = [gene_l[i] + '-' +str(i) for i in range(len(gene_l))]
    label = pd.DataFrame(bdf, index=label.index, columns=label_col)
    
    bdf = np.zeros((label.shape[0],0))
    for gene in gene_l:
        if '+' in gene:
            gene_sl = gene.split('+')
            for i in range(len(gene_sl)):
                if i == 0:
                    df = pred[[gene_sl[i]]].values
                else:
                    df = df + pred[[gene_sl[i]]].values
            bdf = np.concatenate((bdf, df),axis=1)
        else:
            df = pred[[gene]].values
            bdf = np.concatenate((bdf, df),axis=1)
    bdf[bdf>1] = 1
    pred_col = [gene_l[i] + '-' +str(i) for i in range(len(gene_l))]
    pred = pd.DataFrame(bdf, index=label.index, columns=pred_col)
    
    pred = pred[label>=0]
    
    # normal f1
    label_pred = label+pred
    label_pred[label_pred>1]=1
    label_sum = np.sum(label, 0)
    pred_sum = np.sum(pred, 0)
    tp = np.sum((label+ pred) - label_pred,0)
    precision = tp / pred_sum
    recall = tp / label_sum
    F1 = 2 * precision * recall / (precision + recall)
    
    out_put = pd.concat((tp, precision, recall, F1), axis=1)
    out_put.columns = ['TP', 'Precision', 'Recall', 'F1']
    print(out_put)
    out_put.to_csv(output_name, header=True, index=True)
    return out_put


def f1():
    # micro f1
    from sklearn.metrics import f1_score
    f1 = []
    for i in range(pred.shape[1]):
        val_l = [ele for ele in pred.iloc[:,i].tolist() if np.isnan(ele) == False ]
        preds = pred.iloc[val_l,i].tolist()
        labels = label.iloc[val_l,i].tolist()
        f1.append(f1_score(labels, preds, average='macro'))
    f1_df = pd.DataFrame(f1, index=pred.columns, columns=['f1'])

#dr_sen('ggg', fix_mutation=1)
#sta('EMva/Output/S_Aggg1.csv', 'EMva/Output/sta/S_Aggg1.csv', 0, 0.5)

#dr_sen('ttg', fix_mutation=1)
#sta('DrugNormal/Output/S_Attg1.csv', 'EMva/Output/sta/S_Attg1.csv', 0, 0.5)

#dr_sen('ggg_rd', fix_mutation=1)

#top = pd.DataFrame(np.zeros((52,4)))
#for i in range(1,10):
#    out_put = sta('EMva/Output/S_Ad1_rad'+'-'+str(i)+'.csv', 'EMva/Output/sta/S_Ad1_rad'+'-'+str(i)+'.csv', 0, 0.5)
#    top.index=out_put.index
#    top.columns = out_put.columns
#    top = top+out_put
#top = top/10
#top.to_csv('EMva/Output/sta/S_Ad1_rad_mean.csv', header=True, index=True)
