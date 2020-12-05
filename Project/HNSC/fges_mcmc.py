#!/usr/bin/env python
# coding: utf-8
import pandas as pd
import numpy as np
from copy import deepcopy
import json
import sys
import os
import time
import javabridge
from multiprocessing import Process

def create_knowledge(sga_l, A_D, knowledge_path):
    
    A_D["cause gene name"] = A_D.index
    TAD = pd.melt(A_D, id_vars="cause gene name", value_vars=list(A_D.columns[0:-1]),
                  var_name="result gene name", value_name="value")
    forbid_l = TAD[TAD["value"] == 0].iloc[:, [0, 1]].values.tolist()

    print("/knowledge", file=open(knowledge_path, "w"))
    print("addtemporal", file=open(knowledge_path, "a"))
    sga_s = "1 " + " ".join([str(sga) for sga in sga_l])
    print(sga_s, file=open(knowledge_path, "a"))
    deg_s = "2* " + " ".join([str(deg) for deg in A_D.columns[0:-1]])
    print(deg_s, file=open(knowledge_path, "a"))   
    
    print('\n', file=open(knowledge_path, "a"))
    print("forbiddirect", file=open(knowledge_path, "a"))
    for i in forbid_l:
        print(i[0] + " " + i[1], file=open(knowledge_path, 'a'))
    
    print('\n', file=open(knowledge_path, "a"))
    print("requiredirect", file=open(knowledge_path, "a"))
    
def py_causal(S_A, S_Ad, S_D, A_D, sga_l, knowledge_path):
    
    ## data prepration
    sga_l = A_D.index & set(sga_l)
    A_D = A_D.loc[sga_l, np.sum(A_D.loc[sga_l], 0)>=1]
    A_D = A_D.loc[S_A.columns&A_D.index, S_D.columns&A_D.columns]
    S_A = S_A[A_D.index]
    S_Ad = S_Ad[A_D.index]
    S_D = S_D[A_D.columns]

    S_AdD = pd.concat([S_Ad, S_D], 1)

    ## connect to java
    from pycausal.pycausal import pycausal as pc
    pc = pc()
    pc.start_vm(java_max_heap_size="1000M")

    ## generate knowledge
    from pycausal import prior as p
    create_knowledge(sga_l, A_D, knowledge_path)
    prior = p.knowledgeFromFile(knowledge_path)

    ## search
    from pycausal import search as s
    tetrad = s.tetradrunner()

    tetrad.run(dfs=S_AdD, dataType="discrete", priorKnowledge=prior, algoId="fges", 
               scoreId="disc-bic-score", structurePrior=1.0, samplePrior=1.0, maxDegree=20,
               faithfulnessAssumed=True, symmetricFirstStep=True, verbose=True)#, 
               #numberResampling=100, percentResampleSize=0, resamplingEnsemble=1,
               #addOriginalDataset=True,
               #resamplingWithReplacement=True)

    node_l = tetrad.getNodes()
    edge_l = tetrad.getEdges()
    bic = tetrad.getTetradGraph()
    return node_l, edge_l, bic, S_A, S_AdD

def java_causal(S_A, S_Ad, S_D, A_D, sga_l, input_path, out_path, knowledge_path):
    
    ## data prepration
    sga_l = A_D.index & set(sga_l)
    A_D = A_D.loc[sga_l, np.sum(A_D.loc[sga_l], 0)>=1]
    A_D = A_D.loc[S_A.columns&A_D.index, S_D.columns&A_D.columns]
    S_A = S_A[A_D.index]
    S_Ad = S_Ad[A_D.index]
    S_D = S_D[A_D.columns]

    S_AdD = pd.concat([S_Ad, S_D], 1)
    S_AdD.to_csv(input_path, index=False, header=True)
    create_knowledge(sga_l, A_D, knowledge_path)
    print(1)

    os.system("java -jar causal-cmd-1.1.1-jar-with-dependencies.jar --algorithm fges \
        --score disc-bic-score" "--dataset %s"%input_path + "--data-type discrete --delimiter comma"+
        "--knowledge %s"%knowledge_path + "--out %s"%out_path + "--structurePrior 1.0 \
        --samplePrior 1.0 --maxIndegree 3 --maxIndegree 20 --faithfulnessAssumed True \
        --symmetricFirstStep True --verbose True --numberResampling 100 --resampleSize 1000 \
        --resamplingEnsemble 1 --addOriginalDataset True --resamplingWithReplacement True")


def mcmc(file_path, i, j):
    
    exe_path = "./MCMC/inferSGAInNetwork_TDI.exe"
    m_path = " -m " + file_path + "/run%i/completeMatrix.csv" % i
    i_path = " -i " + file_path + "/S_A0.csv"
    e_path = " -e " + file_path + "/run%i/Edge.csv" % i
    o_path = " -o " + file_path + "/run%d/ -x 100" % j
    combine = exe_path + m_path + i_path + e_path + o_path
    p1 = Process(target=os.system, args=(combine,))
    p1.start()
    p1.join()
    
    print("Runing")
    
def fges_mcmc(S_A, S_Ad, S_D, A_D, sga_l, file_path, sys_iter, pp_data=1):
    
    ## data prepration
    if pp_data==1:
        
        if 'sga:' not in S_A.columns[0]:
            S_A.columns = ['sga:'+ele for ele in S_A.columns]
        if 'sga:' not in S_Ad.columns[0]:
            S_Ad.columns = ['sga:'+ele for ele in S_Ad.columns]
        if 'deg:' not in S_D.columns[0]:
            S_D.columns = ['deg:'+ele for ele in S_D.columns] 
        if 'sga:' not in A_D.index[0]:
            A_D.index = ['sga:'+ele for ele in A_D.index]
        if 'deg:' not in A_D.columns[0]:
            A_D.columns = ['deg:'+ele for ele in A_D.columns]
        if 'sga:' not in sga_l[0]:
            sga_l = ["sga:" + ele for ele in sga_l]
        
        if os.path.exists(file_path + "/run0")==False:
            os.makedirs(file_path + "/run0")
    
    ### run FGES + MCMC
    for i in range(sys_iter):
        
        file_l = os.listdir(file_path + "/run%i" % i)
        while "BIC.txt" not in file_l:
            
            knowledge_path = file_path + "/run%i/Knowledge.csv" %i
            node_l, edge_l, bic, S_A_sub, S_AD_sub = py_causal(S_A, S_Ad, S_D, A_D, sga_l, knowledge_path)
            
            # save edges.csv
            edge_split_l = [edge.split(" ") for edge in edge_l if "---" not in edge]
            edge_df = pd.DataFrame(edge_split_l).iloc[:, [0, 2]]
            edge_df.to_csv(file_path + "/run%i/Edge.csv" %i, index=False, header=False)

            # save BIC.txt
            print(bic, file=open(file_path + "/run%i/BIC.txt" % i, "w"))
                        
            # save S_A, S_AdD
            S_A_sub = pd.concat([S_A_sub]*5, ignore_index=False)
            S_A_sub.to_csv(file_path+'/S_A0.csv', header=True, index=False)
            S_AD_sub = pd.concat([S_AD_sub]*5, ignore_index=False)
            S_AD_sub.to_csv(file_path+'/run%s/completeMatrix.csv'%i, header=True, index=False)
            
            # update file_l
            file_l = os.listdir(file_path + "/run%i" % i)
            
        else:
            
            j = i+1
            
            if os.path.exists(file_path + "/run%d" % j)==False:
                os.makedirs(file_path + "/run%d" % j)
                
            next_file_l = os.listdir(file_path + "/run%i" % j)
            while "completeMatrix.csv" not in next_file_l:
                
                mcmc(file_path, i, j)

                # update file_l
                next_file_l = os.listdir(file_path + "/run%i" % j)
                
            else:
                print("NEXT")
