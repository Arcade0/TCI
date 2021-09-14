import pandas as pd

# BRCA
BRCA_sub = pd.read_csv("chunhui/Input/BRCA/Subtype_BRCA.csv",
                       index_col=0,
                       header=0)
sb_l = [0, "Basal", "Her2", "LumA", "LumB", "Normal"]

# BRCA PRO
S_Pi = pd.read_csv("chunhui/Output/BRCA/S_Ps_BRCA_ct.csv",
                   index_col=0,
                   header=0).loc[BRCA_sub.index]
S_Pi_f = ["TCGA-AQ-A04J", "TCGA-AC-A62X", "TCGA-A2-A1FX", "TCGA-A8-A07L"]

# # BRCA DEG
# S_Dci = S_Dc.loc[BRCA_sub.index, A_D.columns & S_Dc.columns]
# S_Dci_f = ["TCGA-A7-A3J0", "TCGA-A2-A0CO", "TCGA-AC-A3W7", "TCGA-D8-A1XL"]

# Survival Data
sur_df = pd.read_table("chunhui/Input/BRCA/BRCA_survival.txt",
                       sep="\t",
                       index_col=1)
par = "OS"
sur_dfs = sur_df[["%s.time" % par, par]].dropna(axis=0, how="any")
