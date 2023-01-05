# -*- coding: utf-8 -*-
"""
"""
import glob 
import numpy as np
import pandas as pd
#%%results loading
path = r'..'
# try:
# dctPdo[intf_] = {}
fish4 = path + '/**/*.npy'
list_of_files = glob.glob(fish4, recursive=True)
# %%
df = pd.DataFrame(columns=['M', 'T', 'S'], index=range(len(list_of_files)))
#%% gathering results
i = 0
for  _ in list_of_files:
    s_ = _.split('\\')[1]
    t_ = _.split('\\')[2].split('_')[1]
    r_ = np.load(_, allow_pickle=True).tolist()
    mse_ = r_['losses_val'].min()
    df.iloc[i] = [mse_, t_, s_ ]
    i+=1
    print(_)
# %% savinf results to csv
# df.to_csv('violin_gf_03.csv')
