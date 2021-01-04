import pandas as pd

# LUAD TCI MUT
S_Ai = pd.read_csv("chunhui/Output/LUAD/S_Ai_clu.csv", header=0, index_col=0)
S_Ai.index = [ele[0:12] for ele in S_Ai.index]
S_Ai_f = ["TCGA-J2-A4AG", "TCGA-69-7760", "TCGA-44-6148" ]

# LUAD TCI DEG
S_Di = pd.read_csv("chunhui/Output/LUAD/S_Di_clu.csv", header=0, index_col=0)
S_Di.index = [ele[0:12] for ele in S_Di.index]
S_Di_f = ["TCGA-35-5375", "TCGA-44-7661", "TCGA-55-7815"] # disc DEG

S_Dci = pd.read_csv("chunhui/Output/LUAD/S_Dci_clu.csv", header=0, index_col=0)
S_Dci.index = [ele[0:12] for ele in S_Dci.index]
S_Dci_f = ["TCGA-49-4486", "TCGA-MP-A4T6", "TCGA-55-7995"]  # ct DEG

# All DEG
# S_Dai = pd.read_csv("chunhui/Output/LUAD/S_Dai_clu.csv", header=0, index_col=0)
# S_Dai.index = [ele[0:12] for ele in S_Dai.index]
# S_Dai_f = ["TCGA-QK-A6V9", "TCGA-CN-5358", "TCGA-CV-7248"]

# S_Dcai = pd.read_csv("chunhui/Output/LUAD/S_Dcai_clu.csv", header=0, index_col=0)
# S_Dcai.index = [ele[0:12] for ele in S_Dcai.index]
# S_Dcai_f = ["TCGA-CV-A6JT", "TCGA-KUA6-H8", "TCGA-CN-A6V3"]

# LUAD TCI PRO
S_Pi = pd.read_csv("chunhui/Output/LUAD/S_Pi_clu.csv", header=0, index_col=0)
S_Pi.index = [ele[0:12] for ele in S_Pi.index]
S_Pi_f = ["TCGA-91-6828", "TCGA-50-5933", "TCGA-55-7815"] # disc PRO

S_Pci = pd.read_csv("chunhui/Output/LUAD/S_Pci_clu.csv", header=0, index_col=0)
S_Pci.index = [ele[0:12] for ele in S_Pci.index]
S_Pci_f = ["TCGA-86-8359", "TCGA-73-7499", "TCGA-55-7994"]  # ct PRO

# Survival Data
sur_df = pd.read_table("chunhui/Input/LUAD/LUAD_survival.txt", sep="\t", index_col=1)
par = "OS"
sur_dfs = sur_df[["%s.time" % par, par]].dropna(axis=0, how="any")