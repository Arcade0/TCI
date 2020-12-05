import numpy as np
import pandas as pd
import os
import json

dt = pd.read_excel('DrugNormal/Input/Screened_Compounds.xlsx', sheetname='Sheet1')
dn = []
dna = []
for i in range(dt.shape[0]):
    if 'PI3' in dt.iloc[i, 3]:
        dn.append(str(dt.iloc[i, 0]))
        dna.append(dt.iloc[i, :])
dfna = pd.DataFrame(dna)
dfna.to_csv('PI3K_Drug.csv', index=True, header=True)

dr = pd.read_csv('DrugNormal/Input/GDSC drug response.csv', header=0, index_col=0)
pi3k = dr.loc[:,dn]
pi3kll = []
for j in range(pi3k.shape[1]):
    pi3kk = []
    pi3kv = []
    for k in range(pi3k.shape[0]):
        if pi3k.iloc[k, j] == True:
            pi3kk.append(pi3k.index[k])
            pi3kv.append(1)
        if pi3k.iloc[k, j] == False:
            pi3kk.append(pi3k.index[k])
            pi3kv.append(0)
    pi3kki = [str(pi3kk[i]) for i in range(len(pi3kk))]
    pi3kll.append(dict(zip(pi3kki, pi3kv)))

dni = [str(dn[i]) for i in range(len(dn))]
pi3kr = dict(zip(dni, pi3kll))
#print(pi3kr)

with open('PI3K_RESPONSE.json', 'w') as js:
    json.dump(pi3kr, js)


mu = pd.read_csv('DrugNormal/Input/GDSC mutation.csv', header=0, index_col=0)
mug = []
for i in range(mu.shape[0]):
    if 'PIK3' in mu.index[i]:
        mug.append(mu.index[i])

pi3kmu = mu.loc[mug,:].T
print(pi3kmu)
pi3kmu.to_csv('PI3K_Mutation.csv', index=True, header=True)

