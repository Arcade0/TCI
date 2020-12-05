import pandas as pd
import numpy as np
import sys
import os
from copy import deepcopy
import time
import javabridge



def mk_dir(file_path):

    folder = os.path.exists(file_path)
    if not folder:
        os. makedirs(file_path)


def EM(S_Am, S_Dm, A_D, fix_mutation=0, e=1e-5, cut_value=0.85, ite=10000, cover_thres=1e-6):

    S_Ad = S_Am.values
    S_Dd = S_Dm.values
    A_Dd = A_D.values
    pA1_mean_l = [1]

    for i in range(ite):

        pA1 = (np.sum(S_Ad, 0)[:, None]+1)/(S_Ad.shape[0]+2)
        pD1A1 = (np.dot(S_Ad.T, S_Dd) + 1)/(np.sum(S_Ad, 0)[:, None]+2)
        pD1A0 = (np.dot(1-S_Ad.T, S_Dd)+1)/(np.sum(1-S_Ad, 0)[:, None]+2)

        logpA1 = np.log(pA1 + e)
        logpA0 = np.log(1-pA1 + e)
        logpD1A1 = np.log(pD1A1 + e) * A_Dd
        logpD0A1 = np.log(1-pD1A1 + e) * A_Dd
        logpD1A0 = np.log(pD1A0 + e) * A_Dd
        logpD0A0 = np.log(1-pD1A0 + e) * A_Dd

        logpA1D = np.dot(S_Dd, logpD1A1.T) + \
            np.dot(1 - S_Dd, logpD0A1.T) + logpA1.T
        logpA0D = np.dot(S_Dd, logpD1A0.T) + \
            np.dot(1 - S_Dd, logpD0A0.T) + logpA0.T
        S_Ad = 1 / (1 + np.exp(logpA0D - logpA1D))

        if fix_mutation == 1:
            S_Ad = S_Ad+S_Am
            S_Ad[S_Ad > 1] = 1
        else:
            S_Ad = S_Ad

        pA1_mean_l.append(np.mean(pA1))
        if (pA1_mean_l[i] - pA1_mean_l[i-1])**2 < cover_thres:
            break

    S_Ad0 = pd.DataFrame(S_Ad, index=S_Am.index, columns=S_Am.columns)
    S_Ad1 = deepcopy(S_Ad0)
    S_Ad1[S_Ad1 > cut_value] = 1
    S_Ad1[S_Ad1 <= cut_value] = 0

    para = [logpA1, logpA0, logpD1A1, logpD0A1, logpD1A0, logpD0A0]
    return S_Ad0, S_Ad1, para


def create_TAD(df):

    df['cause gene name'] = df.index
    TAD = pd.melt(df, id_vars='cause gene name',
                  value_vars=list(df.columns[0:-1]),
                  var_name='result gene name', value_name='value')
    TAD1 = TAD[TAD['value'] == 1]
    TAD0 = TAD[TAD['value'] == 0]

    return TAD, TAD1, TAD0


def combine_S_AD(S_Am, S_Dm, TAD1, SGA_l):

    S_ADm = pd.concat((S_Am, S_Dm), axis=1)

    nTAD1 = TAD1[TAD1['cause gene name'].isin(SGA_l)]
    DEG_l = list(np.unique(nTAD1['result gene name']))

    SGA_l = [ele for ele in SGA_l if ele in S_Am.columns]
    DEG_l = [ele.replace('.', '-') for ele in DEG_l]
    nS_Am = S_Am[SGA_l]
    nS_ADm = S_ADm[SGA_l+DEG_l]

    return nS_Am, nS_ADm


def create_knowledge(SGA, SGA_l, A_D):

    A_D['cause gene name'] = A_D.index
    TAD = pd.melt(A_D, id_vars='cause gene name', value_vars=list(A_D.columns[0:-1]), var_name='result gene name', value_name='value')
    TAD1 = TAD[TAD['value'] == 1]
    TAD0 = TAD[TAD['value'] == 0]

    SGA.columns = ['cause gene name']
    TAD1_SGA = pd.merge(SGA, TAD1, 'inner').iloc[:, [0, 1]]
    TAD0_SGA = pd.merge(SGA, TAD0, 'inner').iloc[:, [0, 1]]
    forbid_l = TAD0_SGA.values.tolist()

    return forbid_l


def fges_stem(file_path, sys_iter, SGA_l, A_D):

    BIC_l = [float(0)]

    SGA = pd.DataFrame(SGA_l)
    SGA.columns = ['cause gene name']
    A_D_i = A_D

    for i in range(sys_iter):
        print(i)
        file_l = os.listdir(file_path + '/Output/run%i' % i)
        while 'completeMatrixn.csv' not in file_l:
            df_name = file_path + '/Output/run%i/completeMatrix.csv' % i
            df = pd.read_csv(df_name, header=0, index_col=None)

            from pycausal.pycausal import pycausal as pc
            pc = pc()
            pc.start_vm(java_max_heap_size='5000M')

            from pycausal import prior as p
            # get knowledge from knowledge file
            # prior = p.knowledgeFromFile(file_path + '/Input/Knowledge')
            # get knowledge from DEG and SGA list
            DEG_l = [x for x in df.columns if x not in SGA_l]
            A_D_i = A_D_i[DEG_l]
            forbid = create_knowledge(SGA, SGA_l, A_D_i)
            temporal = [SGA_l, p.ForbiddenWithin(DEG_l)]
            prior = p.knowledge(forbiddirect=forbid, addtemporal=temporal)

            from pycausal import search as s
            tetrad = s.tetradrunner()
            tetrad.getAlgorithmParameters(algoId='fges', scoreId='bdeu')

            tetrad.run(algoId='fges', dfs=df, scoreId='bdeu', priorKnowledge=prior, dataType='discrete', structurePrior=1.0, samplePrior=1.0,
                       maxDegree=1000, faithfulnessAssumed=True, verbose=True)  # , numberResampling=10, resamplingEnsemble=1, addOriginalDataset=True)

            # save edges.csv
            node_l = tetrad.getNodes()
            edge_l = tetrad.getEdges()
            edge_split_l = []
            for edge in edge_l:
                if '---' in edge:
                    edge_n = edge.split(' ')
                    if np.sum(df[edge.split(' ')[0]]) > np.sum(df[edge.split(' ')[2]]):
                        edge_n.reverse()
                    else:
                        edge_n = edge_n
                    edge_split_l.append(edge_n)
                else:
                    edge_split_l.append(edge.split(' '))

            #edge_split_l = [edge.split(' ') for edge in edge_l if '---' not in edge]

            edge_df = pd.DataFrame(edge_split_l).iloc[:, [0, 2]]
            edge_df.to_csv(file_path + '/Output/run%i/Edge.csv' %i, index=False, header=False)

            # save completeMatrixn.csv
            new_df = df.loc[:, node_l]
            new_df.to_csv(file_path + '/Output/run%i/completeMatrixn.csv' %i, index=False, header=True)

            # save BIC.txt
            print(tetrad.getTetradGraph(), file=open(file_path + '/Output/run%i/BIC.txt' % i, 'a'))

            file_l = os.listdir(file_path + '/Output/run%i' % i)

        else:
            # save BIC which used to verify convergency
            with open(file_path+'/Output/run%i/BIC.txt' % i, 'r') as BIC_txt:
                for line in BIC_txt:
                    if 'BIC: -' in line:
                        BIC_l.append(float(line[5:-1]))

            j = i+1
            mk_dir(file_path + '/Output/run%d' % j)
            next_file_l = os.listdir(file_path + '/Output/run%i' % j)
            while 'completeMatrix.csv' not in next_file_l:
                exe_path = './inferSGAInNetwork_TDI.exe'
                m_path = ' -m ' + file_path + '/Output/run%i/completeMatrixn.csv' % i
                i_path = ' -i ' + file_path + '/Input/S_A0.csv'
                e_path = ' -e ' + file_path + '/Output/run%i/Edge.csv' % i
                o_path = ' -o ' + file_path + '/Output/run%d/ -x 50' % j
                combine = exe_path + m_path + i_path + e_path + o_path
                os.system(combine)
                time.sleep(20)
                next_file_l = os.listdir(file_path + '/Output/run%i' % j)
            else:
                pd.DataFrame(BIC_l).to_csv(file_path+'/Output/BIC.csv', index=False, header=False)


RTK_l = ['EGFR', 'ERBB2', 'ERBB4', 'MET', 'PDGFRA', 'FGFR1', 'FGFR2', 'FGFR3', 'FGFR4', 'KIT', 'IGF1R', 'RET', 'ROS1', 'ALK', 'FLT3', 'NTRK1-3', 'JAK2', 'CBL', 'ERRFI1','ABL1', 'SOS1', 'NF1', 'RASA1', 'PTPN11', 'KRAS', 'HRAS', 'NRAS', 'RIT1', 'ARAF', 'BRAF', 'RAF1', 'RAC1', 'MAPK1', 'MAP2K1', 'MAP2K2']
PI3K_l = ['PTEN', 'PIK3R2', 'PIK3R1', 'PIK3R2', 'PIK3CA', 'PIK3CB', 'INPP4B',
          'AKT1', 'AKT2', 'AKT3', 'TSC1', 'TSC2', 'RHEB', 'STK11', 'MTOR', 'RICTOR', 'RPTOR']
RB1_l = ['CDKN1A', 'CDKN1B', 'CDKN2A', 'CDKN2B', 'CDKN2C', 'CCNE',
         'CCND1', 'CCND2', 'CCND3', 'CDK2', 'CDK4', 'CDK6', 'RB1', 'E2F1', 'E2F3']
RBl_l = ['CDKN2A', 'CDK4','RB1','E2F3']
NOTCH_l = ['FBXW7','NOTCH1','EP300','CREBBP']


RB1_l = ['sga:'+ele for ele in RB1_l]
RTK_l = ['sga:'+ele for ele in RTK_l]
PI3K_l = ['sga:'+ele for ele in PI3K_l]
SM_l = ['sga:'+str(i) for i in range(7)]
RBl_l = ['sga:'+ ele for ele in RBl_l]

SGA_d = {'RB1': RB1_l, 'RTK': RTK_l, 'PI3K': PI3K_l, 'SM':SM_l, 'RBl':RBl_l, 'NOTCH':NOTCH_l}

#####################################################################################################################################################################

S_Am = pd.read_csv(str(sys.argv[1]), header=0, index_col=0)
S_Dm = pd.read_csv(str(sys.argv[2]), header=0, index_col=0)
A_D = pd.read_csv(str(sys.argv[3]), header=0, index_col=0)

if str(sys.argv[4]) == 'EM':
    S_Ad0, S_Ad1, para = EM(S_Am, S_Dm, A_D, fix_mutation=0, e=1e-5, cut_value=0.85, ite=10000, cover_thres=1e-6)
    S_Ad1.to_csv(str(sys.argv[5]), header=True, index=True)

if str(sys.argv[4]) == 'noEM':
    S_Ad1 = pd.read_csv(str(sys.argv[5]),header=0, index_col=0)

if str(sys.argv[4]) == '10EM':
    S_Ad0, S_Ad1 = EM10fold(S_Am, S_Dm, A_D, A_D_l, fix_mutation=0,
                            e=1e-5, t=0.85, ite=10000, cover_thres=1e-6)



TAD, TAD1, TAD0 = create_TAD(A_D)
SGA_l = SGA_d[str(sys.argv[6])]
nS_Am, nS_ADm = combine_S_AD(S_Am, S_Dm, TAD1, SGA_l)
nS_Ad, nS_ADd = combine_S_AD(S_Ad1, S_Dm, TAD1, SGA_l)

input_path = str(sys.argv[7]) + '/Input'
output_path = str(sys.argv[7]) + '/Output'
mk_dir(input_path)
mk_dir(output_path+'/run0')
nS_Am.to_csv(input_path+'/S_A0.csv', header=True, index=False)
nS_ADd.to_csv(output_path+'/run0'+'/completeMatrix.csv', header=True, index=False)

fges_stem(str(sys.argv[7]), int(sys.argv[8]), SGA_l, A_D)

