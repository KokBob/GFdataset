# -*- coding: utf-8 -*-
"""

"""
# from pathlib import Path
import numpy as np
import glob
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from graph_reduction import resco


p_0 = r'./results_gred/'
l_p = glob.glob(f'{p_0}*.npy')
file_ = l_p[1]

rc = resco(file_)
dfs = rc.get_dfres_stacked('sage')
dfg = rc.get_dfres_stacked()

# %%
ares, pares = [], []
for _ in l_p:
    file_ = _
    rc_ = resco(file_)
    # dfs = rc_.get_dfres_stacked('sage')
    dfs = rc_.get_dfres_stacked('gcn')
    ar = rc_.to_res_array()
    ares.append(ar)
    pares.append(rc_.df_res)
# %%
df_res = pd.DataFrame(ares,)
# design aby se to nastackovalo vedle sebe
gres = []
names = []
for _ in l_p:
    file_ = _
    rc_ = resco(file_)
    dfs = rc_.get_dfres_stacked('sage')
    # dfs = rc_.get_dfres_stacked('gcn')
    ar = rc_.to_res_array()
    ares.append(ar)
    pares.append(rc_.df_res)
    g_ = dfs.groupby('mt')
    g_d = g_.describe()
    # percentiles=None
    df_acc = g_d['acc'][['mean','std']]
    gres.append(df_acc)
    ds_name = rc_.dataset_name
    names.append(ds_name)
names = [val for val in names for _ in (0, 1)]
# names = [val for val in method_name for _ in (0, 1)]
df_fres = pd.concat(gres,axis=1)
# ltx = (df_agg.to_latex(index=True,
#                   formatters={"name": str.upper},
#                   float_format="{:.2f}".format,))  
# %%
al = [names,['mean', 'std']*4] # 4 datasety
tuples = list(zip(*al))
index = pd.MultiIndex.from_tuples(tuples)
df_fres.columns = pd.MultiIndex.from_tuples(tuples, names=['Caps','Lower'])
df_fres_T = df_fres.T
# %%
# suppress scientific notation by setting float_format
pd.options.display.float_format = '{:.3f}'.format

# display the dataframe without scientific notation
print(df_fres_T)
# %%
def format_tex(float_number):
    exponent = np.floor(np.log10(float_number))
    mantissa = float_number/10**exponent
    mantissa_format = str(mantissa)[0:3]
    return "${0}\times10^{{{1}}}$"\
           .format(mantissa_format, str(int(exponent)))
           
# Select mean and std columns
df_acc = g_d['acc'][['mean','std']]
df_acc_summary = g_d['acc'][['mean', 'std']]
# Merge mean and std into a single DataFrame
df_acc_mean_std = df_acc_summary.reset_index()
ltx = (df_agg.to_latex(index=True,
                  formatters={"name": str.upper},
                  float_format="{:.2f}".format,))  
