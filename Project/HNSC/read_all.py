import pandas as pd
# Pan Can
S_Dc = pd.read_csv("../TCI/EM/Input/TCGA expression.csv",
                   index_col=0,
                   header=0)

S_D = pd.read_csv('../TCI/TCGA_info/Input/PanCancer13tts.DEGmatrix.4TCI.csv',
                  index_col=0,
                  header=0)

S_A = pd.read_csv('../TCI/TCGA_info/Input/PanCancer13tts.SGAmatrix.4TCI.csv',
                  index_col=0,
                  header=0)

A_Do = pd.read_csv("../TCI/EM/A_Do.csv", index_col=0, header=0)
