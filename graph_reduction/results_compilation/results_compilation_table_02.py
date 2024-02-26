# -*- coding: utf-8 -*-
"""
refs
[1];https://saturncloud.io/blog/how-to-set-decimal-precision-of-a-pandas-dataframe-column-with-decimal-datatype/
"""
# from pathlib import Path
import numpy as np
import glob
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from graph_reduction import resco
import utils_tab as ut

p_0 = r'./results_gred/'
l_p = glob.glob(f'{p_0}*.npy')
file_ = l_p[1]
rc = resco(file_)
dfs = rc.get_dfres_stacked('sage')
dfg = rc.get_dfres_stacked()

utgcn = ut.buo(l_p)
# df_gcn = utgcn.get_results_dataframe_compiled()
# df_gcn = utgcn.get_full_results() # test = utgcn.gres
dfbig = utgcn.get_big_full()
# %%
df = dfbig
df_ = df.loc['Beam2D', 'std']
# %%
from decimal import Decimal, ROUND_HALF_UP
# fls = df_.values
# exponent = np.floor(np.log10(df_)).astype(str) + 'e'
exponent = np.floor(np.log10(df_))
# mantissa = 

df = dfbig
df_ = df.loc['Beam2D', 'mean']
exponent = np.floor(np.log10(df_))
# es =  exponent
sd = 'E' + exponent.astype(str)
# pd.options.display.float_format = '{:.1f}'.format # nefunguje
mantissa = df_/10**exponent
#%% prehled  metod pro decimalovani [1]
mnt0= mantissa.apply(lambda x: round(x, 2))
mnt0sf= mnt0.apply(lambda x: f'{x:.2f}') # reduntante k metode round 
#%% prehled  metod pro decimalovani [1]
# mnt1 = mantissa.apply(lambda x: Decimal(x).quantize(2)).astype('float64')
# mnt2 = mantissa.apply(lambda x: Decimal(x).quantize(3)).astype('float64')
# Mnt = pd.DataFrame([mnt0,mnt1,mnt2], columns=['r0', 'D2', 'D3'])
# %% exponent funguje jak ma 
# exponent.apply(lambda x: Decimal(x).quantize(1))
# %%
# for col in df.columns:
    # if isinstance(df[col].iloc[0], Decimal):
        # df[col] = df[col].apply(lambda x: round(x, 2))
# mn   = mantissa.astype(str)
# mnt0s   = mantissa.astype(int).astype('str')

exp0 = 'E' + exponent.astype(int).astype(str)
d = mnt0sf + exp0
# pd.options.display.float_format = '{:.2f}'.format
# exponent = np.floor(np.log10(fls))
# mantissa = fls/10**exponent
# mantissa_format = str(mantissa)[0:3]
# s = "${0}\times10^{{{1}}}$".format(mantissa_format, str(int(exponent)))
# %%
# suppress scientific notation by setting float_format
# pd.options.display.float_format = '{:.3f}'.format
# display the dataframe without scientific notation
# print(df.T)
# %%
# df.loc[:, 'std']
# cols = dfbig.columns
