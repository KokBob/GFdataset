# -*- coding: utf-8 -*-
"""
refs
[1];https://saturncloud.io/blog/how-to-set-decimal-precision-of-a-pandas-dataframe-column-with-decimal-datatype/
"""
import glob
import pandas as pd
from graph_reduction import resco
import utils_tab as ut

p_0 = r'./results_gred/'
l_p = glob.glob(f'{p_0}*.npy')
file_ = l_p[1]
rc = resco(file_)
utgcn = ut.buo(l_p)
utsg =  ut.buo(l_p, framework_method_ = 'sage')

dfc = utgcn.get_ltx_preped()
dfs = utsg.get_ltx_preped()
ltx_c = (dfc.T.to_latex(index=True,
                  formatters={"name": str.upper},
                  float_format="{:.2f}".format,))  
ltx_s = (dfc.T.to_latex(index=True,
                  formatters={"name": str.upper},
                  float_format="{:.2f}".format,)) 
# %% tohle je hezky ale zatim to nedam 
# dfcT = dfc.T
# arrays = [
    # [dfcT.index ],
    # ["SAGE"] * len(dfcT.index),
# ]
# tuples = list(zip(*arrays))
# index = pd.MultiIndex.from_tuples(tuples, names=["first", "second"])
# dfc['Framework'] = 'GCN'
# dfs['Framework'] = 'SAGE'
# df_full = pd.concat([dfc,dfs], axis = 0)
# ltx_F = (df_full.T.to_latex(index=True,
#                   formatters={"name": str.upper},
#                   float_format="{:.2f}".format,))  
