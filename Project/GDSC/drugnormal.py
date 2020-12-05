#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
from copy import deepcopy


def dircrete_cellline():
    ttc = pd.read_csv("DrugNormal/Input/GDSC tissue type annotation.csv", header=0, index_col=0)
    tt = pd.read_csv('DrugNormal/Input/Subtypenumbers.csv')
    tt.columns.values[0] = 'TCGA Label'
    ttcn = ttc.merge(tt)
    ttcn.to_csv('DrugNormal/Input/ttcn.csv', header=True, index=True)
    
    S_D = pd.read_csv('DrugNormal/Input/PanCancer13tts.DEGmatrix.4TCI.csv', header=0, index_col=0)
    TCGAexp = pd.read_csv('DrugNormal/Input/TCGA expression.csv', header=0, index_col=0)
    GDSCexp = pd.read_csv('DrugNormal/Input/GDSC expression.csv', header=0, index_col=0)
    
    nTCGAexp_idx = TCGAexp.index & S_D.index
    nTCGAexp_col = TCGAexp.columns & S_D.columns
    nTCGAexp = TCGAexp.loc[nTCGAexp_idx, nTCGAexp_col]
    nS_D = S_D.loc[nTCGAexp_idx, nTCGAexp_col]
    
    nGDSCexp_col = GDSCexp.columns & S_D.columns
    nGDSCexp = GDSCexp[nGDSCexp_col]
    
    thres0 = nTCGAexp[(1 - nS_D) > 0]
    
    norm_mean = np.mean(thres0, 0)
    norm_std = np.std(thres0, 0)
    norm_min = norm_mean - 1.96 * norm_std
    norm_max = norm_mean + 1.95 * norm_std
    
    minGDSCexp = nGDSCexp - norm_min
    minGDSCexp = (minGDSCexp<0)
    maxGDSCexp = nGDSCexp - norm_max
    maxGDSCexp = (maxGDSCexp>0)
    nGDSCexp = maxGDSCexp + minGDSCexp
    
    nGDSCexpt_idx = nnGDSCexp.index & ttcn.iloc[:, 0].tolist()
    nGDSCexpt = nGDSCexp.loc[nGDSCexpt_idx, :]
    nGDSCexp.to_csv('DrugNormal/Output/nGDSCexp.csv', header=True, index=True)
    nGDSCexpt.to_csv('DrugNormal/Output/nGDSCexpt.csv', header=True, index=True)



def split_target():
    
    GDSCdrug = pd.read_excel("DrugNormal/Input/Screened_Compounds.xlsx")
    Y_A = GDSCdrug.iloc[:, [0, 3]]
    drop_l  = []
    for i in range(Y_A.shape[0]):
        A_l = Y_A.iloc[i, 1].replace(' ', '').split(',')
        if len(A_l) > 1:
            drop_l.append(i)
            for j in range(len(A_l)):
                b = A_l[j]+str(i)
                Y_A.loc[b] = [Y_A.iloc[i,0], A_l[j]]
    Y_A = Y_A.drop(Y_A.index[drop_l])
    Y_A.to_csv('DrugNormal/Output/Y_A.csv', header=True, index=False)
    
def drug_sga():
    A_D = pd.read_csv('DrugNormal/Input/A_D.csv', header=0, index_col=0)
    GDSCmut = pd.read_csv('DrugNormal/Input/GDSC mutation.csv', header=0, index_col=0)
    GDSCRe = pd.read_csv('DrugNormal/Input/GDSC drug response.csv', header=0, index_col=0)
    Y_A = pd.read_csv('DrugNormal/Output/Y_A.csv', header=0, index_col=None)

    
    nY_A_idx = [Y_A.iloc[:,1].tolist().index(idx) for idx in A_D.index if idx in Y_A.iloc[:,1].tolist()]
    
    nY_A = Y_A.loc[nY_A_idx,:]
    
    nGDSCRe_col = GDSCRe.columns & nY_A.iloc[:,0].tolist()
    nGDSCRe = GDSCRe.loc[:, nGDSCRe_col]
    
    Y_l = nY_A.iloc[:, 0].tolist()
    Y_l = [str(ele) for ele in Y_l]
    A_l = nY_A.iloc[:, 1].tolist()
    del_l = []
    
    for i in range(nGDSCRe.shape[1]):
        print(i)
        a = Y_l.index(nGDSCRe.columns[i])
        print(a)
        over = [A_l[a]] & nGDSCRe.columns
        if len(over)>0:
            b = list(nGDSCRe.columns).index(A_l[a])
            nGDSCRe.iloc[:,b] = nGDSCRe.iloc[:,i] + nGDSCRe.iloc[:,b]
            del_l.append(i)
        else:
            nGDSCRe.columns.values[i] = A_l[a]
            
    nGDSCRe = nGDSCRe.drop(nGDSCRe.columns[del_l])
    
    ttcn = pd.read_csv('DrugNormal/Input/ttcn.csv', header=0, index_col=0)
    nGDSCRe.to_csv('DrugNormal/Output/nGDSCRe.csv', header=True, index=True)
    nGDSCRet_idx = nGDSCRe.index & ttcn.iloc[:,0].tolist()
    nGDSCRe.loc[nGDSCRet_idx,:].to_csv('DrugNormal/Output/nGDSCRen.csv', header=True, index=True)
    
    nGDSCmut_idx = [idx for idx in GDSCmut.index if idx in nGDSCRe.columns]
    nGDSCmut = GDSCmut.loc[nGDSCmut_idx,:].T
    nGDSCmut.to_csv('DrugNormal/Output/nGDSCmut.csv', header=True, index=True)
    nGDSCmutt_idx = nGDSCmut.index & ttcn.iloc[:,0].tolist()
    nGDSCmut.loc[nGDSCmutt_idx,:].to_csv('DrugNormal/Output/nGDSCmutn.csv', header=True, index=True)

def gene_l():
    gene_l = ['AKT1',
              'AR','ATM',
              'BRAF', 'RAF1', 'PDGFRA','KIT','KDR','CDK4',
              'ERBB2', 'EGFR','FGFR1','FGFR3',
              'MDM4', 'TP53',
              'TOP2A',
              'PPM1D',
              'PIK3CA',
              'MTOR']
        
    for gene in gene_l:
        Y_l = []
        col_l = []
        for i in range(GDSCdrug.shape[0]):
            A_l = GDSCdrug.iloc[i, 3]
            if gene in A_l:
                print(gene + '_' + A_l)
                Y_l.append(str(GDSCdrug.iloc[i, 0]))
                col_l.append(str(GDSCdrug.iloc[i, 0])+'-'+GDSCdrug.iloc[i, 1]+':'+GDSCdrug.iloc[i, 3])
        df = GDSCRe.loc[:,Y_l]
        df.columns = col_l
        df.to_csv('DrugNormal/Output/gene/'+gene+'.csv',header=True, index=True)
        
    ttcn = pd.read_csv('DrugNormal/Input/ttcn.csv', header=0, index_col=0)
    GDSCmut = pd.read_csv('DrugNormal/Input/GDSC mutation.csv', header=0, index_col=0)
    nGDSCmut = GDSCmut.loc[gene_l, :].T
    nGDSCmut.to_csv('DrugNormal/Output/nnGDSCmut.csv', header=True, index=True)
    nGDSCmutt_idx = nGDSCmut.index & ttcn.iloc[:, 0].tolist()
    nGDSCmut.loc[nGDSCmutt_idx, :].to_csv('DrugNormal/Output/nnGDSCmutn.csv', header=True, index=True)
    



