import numpy as np
import pandas as pd
import random
import os

def mk_dir(file_path):

    folder = os.path.exists(file_path)
    if not folder:
        os. makedirs(file_path)
def create_random(input_name, output_name, cut):

    df = pd.read_csv(input_name, header=0, index_col=0)

    rand_df = np.random.rand(df.shape[0], df.shape[1])
    rand_df[rand_df > cut] = 1
    rand_df[rand_df <= cut] = 0
    rand_df = pd.DataFrame(rand_df, index=df.index, columns=df.columns)

    rand_df.to_csv(output_name, index=True, header=True)
    
def create_portion_random(input_name, out_put, fil_name, re, source, partial=1):

    mk_dir(out_put)
    df = pd.read_csv(input_name, header=0, index_col=0).T
    n_ele_l = []
    nn_ele_l = []
    for i in range(df.shape[0]):
        
        n_ele_l.append([idx for idx in range(df.shape[1]) if df.iloc[i, idx]==0])
        nn_ele_l.append([idx for idx in range(df.shape[1]) if df.iloc[i, idx]==1])

    for j in range(re):
        r_m = np.zeros((0, df.shape[1]))
        for i in range(df.shape[0]):
            if source=='in':
                rn_ele_l = random.sample(nn_ele_l[i], int(len(nn_ele_l[i])*partial))
            if source=='out':
                if len((nn_ele_l[i])*partial)>len(n_ele_l[i]):
                    rn_ele_l = random.sample(range(df.shape[1]), int(len(nn_ele_l[i])*partial))
                else:
                    rn_ele_l = random.sample(n_ele_l[i], int(len(nn_ele_l[i])*partial))
                
            if source=='all':
                rn_ele_l = random.sample(range(df.shape[1]), int(len(nn_ele_l[i])*partial))
            m_i = np.zeros((1, df.shape[1]))
            m_i[0, rn_ele_l] = 1
            r_m = np.vstack((r_m, m_i))

        r_df = pd.DataFrame(r_m, index=df.index, columns=df.columns).T
        out_put_j=out_put+'/'+fil_name+'_'+str(j)+'.csv'
        r_df.to_csv(out_put_j, index=True, header=True)