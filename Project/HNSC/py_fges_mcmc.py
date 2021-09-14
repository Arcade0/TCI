#!/usr/bin/env python

import pandas as pd
import numpy as np
import os
import time
from copy import deepcopy


def create_knowledge(sga_l_v, sga_deg_v, knowledge_path):
    """Used to create knowledge file for tetrad.

    Args:
        sga_l: A list contains SGAs.
        sga_deg: 2-D df, index are SGAs or gene names,
                 columns are SGAs or gene names, df's values should be 0 or 1.
        knowledge_path: String, file path to save knowledge file.

    Returns:
        Save knowledge file to knowledge_path.
    """
    # avoid changing input file
    sga_l = deepcopy(sga_l_v)
    sga_deg = deepcopy(sga_deg_v)

    sga_deg["cause gene name"] = sga_deg.index
    sga_deg_l = pd.melt(sga_deg,
                        id_vars="cause gene name",
                        value_vars=list(sga_deg.columns[0:-1]),
                        var_name="result gene name",
                        value_name="value")
    forbid_l = sga_deg_l[sga_deg_l["value"] == 0].iloc[:,
                                                       [0, 1]].values.tolist()

    with open(knowledge_path, "w") as f:
        f.write("/knowledge\n" + "forbiddirect\n")

    with open(knowledge_path, "a") as f:
        for i in forbid_l:
            f.write(i[0] + " " + i[1] + "\n")

    sga_s = "1 " + " ".join([str(sga) for sga in sga_l])
    deg_s = "2* " + " ".join([str(deg) for deg in sga_deg.columns[0:-1]])
    with open(knowledge_path, "a") as f:
        f.write("requiredirect\n" + "addtemporal\n" + sga_s + "\n" + deg_s)


def data_pp(sp_sga_v, sp_pro_v, sp_deg_v, sga_deg_v, sga_l_v):
    """Notes for data_pp.

    Args:
        sp_sga: 2-D df, index are sample IDs, columns are SGAs or gene names. df's value is 0 or 1.
        sp_pro: 2-D df, index are sample IDs, columns are PROs or gene names. df's value is 0 or 1.
        sp_deg: 2-D df, index are sample IDs, columns are DEGs or gene names. df's value is 0 or 1.
        sga_deg: 2-D df, index are SGAs or gene names, columns are DEGs or gene names. df's value is 0 or 1.
        sga_l: A list contains SGAs.
        knowledge_path: String, file path to save knowledge file.

    Returns:
        sp_sga: 2-D df, index are sample IDs, columns are SGAs or gene names, used in mcmc.
        sp_pro_deg: 2-D df, index are sample IDs, columns are SGAs and DGEs names, used in mcmc.
    """

    # avoid changing input file
    sp_sga = deepcopy(sp_sga_v)
    sp_pro = deepcopy(sp_pro_v)
    sp_deg = deepcopy(sp_deg_v)
    sga_deg = deepcopy(sga_deg_v)
    sga_l = deepcopy(sga_l_v)

    # data prepration
    sga_l = sga_deg.index & set(sga_l)
    sga_deg = sga_deg.loc[sga_l, np.sum(sga_deg.loc[sga_l], 0) >= 1]
    sga_deg = sga_deg.loc[sp_sga.columns & sga_deg.index,
                          sp_deg.columns & sga_deg.columns]
    sp_sga = sp_sga[sga_deg.index]
    sp_pro = sp_pro[sga_deg.index]
    sp_deg = sp_deg[sga_deg.columns]

    # add pre-add
    if 'sga:' not in sp_sga.columns[0]:
        sp_sga.columns = ['sga:' + ele for ele in sp_sga.columns]
    if 'sga:' not in sp_pro.columns[0]:
        sp_pro.columns = ['sga:' + ele for ele in sp_pro.columns]
    if 'deg:' not in sp_deg.columns[0]:
        sp_deg.columns = ['deg:' + ele for ele in sp_deg.columns]
    if 'sga:' not in sga_deg.index[0]:
        sga_deg.index = ['sga:' + ele for ele in sga_deg.index]
    if 'deg:' not in sga_deg.columns[0]:
        sga_deg.columns = ['deg:' + ele for ele in sga_deg.columns]
    if 'sga:' not in sga_l[0]:
        sga_l = ["sga:" + ele for ele in sga_l]

    sp_pro_deg = pd.concat([sp_pro, sp_deg], 1)

    return sp_sga, sp_pro_deg, sga_l, sga_deg


def py_causal(sp_pro_deg_v,
              knowledge_path,
              bp=False,
              dataType="discrete",
              algoId="fges",
              scoreId="disc-bic-score",
              structurePrior=1.0,
              samplePrior=1.0,
              maxDegree=20,
              faithfulnessAssumed=True,
              symmetricFirstStep=True,
              verbose=True,
              numberResampling=100,
              percentResampleSize=90,
              resamplingEnsemble=1,
              addOriginalDataset=True,
              resamplingWithReplacement=True):
    """Notes for py_causal.

    Args:
        sp_sga_deg: 2-D df, index are sample IDs, columns are SGAs and DGEs names.
                    df's value is 0 or 1.
        bp: bool, using bootstrap. Default is False.

    Returns:
        node_l: A list of contain nodes in network.
        edge_l: A 2-D df contains two columns, source node to target node,
                didn't contain edge without direction.
        bic: fges output, contain nodes, edges, network score.
    """
    # avoid changing input file
    sp_pro_deg = deepcopy(sp_pro_deg_v)

    # connect to java
    from pycausal.pycausal import pycausal as pc
    pc = pc()
    pc.start_vm(java_max_heap_size="1000M")

    # generate knowledge
    from pycausal import prior as p
    prior = p.knowledgeFromFile(knowledge_path)

    # search
    from pycausal import search as s
    tetrad = s.tetradrunner()

    if bp is True:
        tetrad.run(dfs=sp_pro_deg,
                   priorKnowledge=prior,
                   dataType=dataType,
                   algoId=algoId,
                   scoreId=scoreId,
                   structurePrior=structurePrior,
                   samplePrior=samplePrior,
                   maxDegree=maxDegree,
                   faithfulnessAssumed=faithfulnessAssumed,
                   symmetricFirstStep=symmetricFirstStep,
                   verbose=verbose,
                   numberResampling=numberResampling,
                   percentResampleSize=percentResampleSize,
                   resamplingEnsemble=resamplingEnsemble,
                   addOriginalDataset=addOriginalDataset,
                   resamplingWithReplacement=resamplingWithReplacement)
    else:
        tetrad.run(dfs=sp_pro_deg,
                   priorKnowledge=prior,
                   dataType=dataType,
                   algoId=algoId,
                   scoreId=scoreId,
                   structurePrior=structurePrior,
                   samplePrior=samplePrior,
                   maxDegree=maxDegree,
                   faithfulnessAssumed=faithfulnessAssumed,
                   symmetricFirstStep=symmetricFirstStep,
                   verbose=verbose)

    node_l = tetrad.getNodes()
    edge_l = tetrad.getEdges()
    bic = tetrad.getTetradGraph()

    return node_l, edge_l, bic


def java_causal(input_path, out_path, knowledge_path):

    # run in shell
    os.system(
        "java -jar causal-cmd-1.1.3-jar-with-dependencies.jar --algorithm fges \
        --score disc-bic-score --dataset %s --data-type discrete --delimiter comma \
        --knowledge %s --faithfulnessAssumed True --symmetricFirstStep True \
        --verbose True" % (input_path, knowledge_path))


def mcmc(file_path, i, j, mcmc_ite=50):
    """Notes for mcmc.

    Args:
        file_path: String, the file folder used to read input and save output.
        i: int, used in run loop.
        j: int, used in run loop.
        mcmc_ite: The max iteration times of running mcmc. Dafault is 50.

    Returns:
        Save a modified new matrix to file_path folder.
    """

    exe_path = "./MCMC/inferSGAInNetwork_TDI.exe"
    m_path = " -m " + file_path + "/run%i/completeMatrix.csv" % i
    i_path = " -i " + file_path + "/sp_sga.csv"
    e_path = " -e " + file_path + "/run%i/Edge.csv" % i
    o_path = " -o " + file_path + "/run%i/ -x %i" % (j, mcmc_ite)
    combine = exe_path + m_path + i_path + e_path + o_path

    x = os.system(combine)
    time.sleep(20)
    print(x)


def fges_mcmc(sp_sga_v,
              sp_pro_v,
              sp_deg_v,
              sga_deg_v,
              sga_l_v,
              file_path,
              sys_iter=10,
              bp=False,
              dataType="discrete",
              algoId="fges",
              scoreId="disc-bic-score",
              structurePrior=1.0,
              samplePrior=1.0,
              maxDegree=20,
              faithfulnessAssumed=True,
              symmetricFirstStep=True,
              verbose=True,
              numberResampling=100,
              percentResampleSize=90,
              resamplingEnsemble=1,
              addOriginalDataset=True,
              resamplingWithReplacement=True,
              mcmc_ite=50):
    """Notes for py_causal.

    Args:
        sp_sga: 2-D df, index are sample IDs, columns are SGAs or gene names. df's value is 0 or 1.
        sp_pro: 2-D df, index are sample IDs, columns are PROs or gene names. df's value is 0 or 1.
        sp_deg: 2-D df, index are sample IDs, columns are DEGs or gene names. df's value is 0 or 1.
        sga_deg: 2-D df, index are SGAs or gene names, columns are DEGs or gene names. df's value is 0 or 1.
        sga_l: A list contains SGAs.
        file_path: String, folder used to save output.
        sys_iter: Int, The max iteration times of running fges_mcmc loop, Default=10.
        bp: bool, using bootstrap. Default is False.
        mcmc_ite: The max iteration times of running mcmc. Defalut is 50.

    Returns:
        py_causal save node_l, edge_l, bic to current run folder,
        mcmc save modified sp_pro_deg to next folder.

    """

    # avoid changing input file
    sp_sga = deepcopy(sp_sga_v)
    sp_pro = deepcopy(sp_pro_v)
    sp_deg = deepcopy(sp_deg_v)
    sga_deg = deepcopy(sga_deg_v)
    sga_l = deepcopy(sga_l_v)

    if os.path.exists(file_path + "/run0") is False:
        os.makedirs(file_path + "/run0")

    # data prepration
    sp_sga_sub, sp_pro_deg_sub, sga_l_sub, sga_deg_sub = data_pp(
        sp_sga, sp_pro, sp_deg, sga_deg, sga_l)

    sp_sga_sub = pd.concat([sp_sga_sub] * 5, ignore_index=False)
    sp_sga_sub.to_csv(file_path + '/sp_sga.csv', header=True, index=False)

    knowledge_path = file_path + "/Knowledge.txt"
    create_knowledge(sga_l_sub, sga_deg_sub, knowledge_path)

    sp_pro_deg_sub = pd.concat([sp_pro_deg_sub] * 5, ignore_index=False)
    sp_pro_deg_sub.to_csv(file_path + '/run0/completeMatrix.csv',
                          header=True,
                          index=False)

    # run FGES + MCMC
    for i in range(sys_iter):

        file_l = os.listdir(file_path + "/run%i" % i)
        while "BIC.txt" not in file_l:

            sp_pro_deg_sub = pd.read_csv(file_path +
                                         '/run%i/completeMatrix.csv' % i,
                                         header=0,
                                         index_col=None)

            node_l, edge_l, bic = py_causal(
                sp_pro_deg_v=sp_pro_deg_sub,
                knowledge_path=knowledge_path,
                bp=bp,
                dataType=dataType,
                algoId=algoId,
                scoreId=scoreId,
                structurePrior=structurePrior,
                samplePrior=samplePrior,
                maxDegree=maxDegree,
                faithfulnessAssumed=faithfulnessAssumed,
                symmetricFirstStep=symmetricFirstStep,
                verbose=verbose,
                numberResampling=numberResampling,
                percentResampleSize=percentResampleSize,
                resamplingEnsemble=resamplingEnsemble,
                addOriginalDataset=addOriginalDataset,
                resamplingWithReplacement=resamplingWithReplacement)

            # save edges.csv
            edge_split_l = [
                edge.split(" ") for edge in edge_l if "---" not in edge
            ]
            edge_df = pd.DataFrame(edge_split_l).iloc[:, [0, 2]]
            edge_df.to_csv(file_path + "/run%i/Edge.csv" % i,
                           index=False,
                           header=False)

            # save BIC.txt
            bic_f = open(file_path + "/run%i/BIC.txt" % i, "w")
            print(bic, file=bic_f)
            bic_f.close()

            file_l = os.listdir(file_path + "/run%i" % i)

        else:

            j = i + 1

            if os.path.exists(file_path + "/run%d" % j) is False:
                os.makedirs(file_path + "/run%d" % j)

            next_file_l = os.listdir(file_path + "/run%i" % j)
            while "completeMatrix.csv" not in next_file_l:

                mcmc(file_path, i, j, mcmc_ite=mcmc_ite)
                next_file_l = os.listdir(file_path + "/run%i" % j)

            else:
                print("NEXT")
