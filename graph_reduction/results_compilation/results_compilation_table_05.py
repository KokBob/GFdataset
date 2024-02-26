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
from decimal import Decimal, ROUND_HALF_UP

p_0 = r'./results_gred/'
l_p = glob.glob(f'{p_0}*.npy')
file_ = l_p[1]
rc = resco(file_)
dfg = rc.get_dfres_stacked('gcn')
dfs = rc.get_dfres_stacked('sage')


utgcn = ut.buo(l_p)
utsg =  ut.buo(l_p, framework_method_ = 'sage')
dfgcn = utgcn.get_big_full()
dfsg =  utsg.get_big_full()
# test: pokud je to nenulova matice tak fajn 
# print(dfgcn-dfsg)
def mantisator(serie_, precision = 2):
    exponent = np.floor(np.log10(serie_))
    mantissa = serie_/10**exponent
    mnt = mantissa.apply(lambda x: round(x, precision)) 
    mnts = mnt.apply(lambda x: f'{x:.1f}') # nebo: f'{x:.2f}') etc
    exp0 = 'E' + exponent.astype(int).astype(str)
    return mnts + exp0

dfc = dfgcn.apply(lambda x: mantisator(x)).T
dfs = dfsg.apply(lambda x: mantisator(x)).T
df_full = pd.concat([dfc,dfs], axis = 0)
ltx = (dfc.T.to_latex(index=True,
                  formatters={"name": str.upper},
                  float_format="{:.2f}".format,))  


