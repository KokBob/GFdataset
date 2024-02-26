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
dfs = rc.get_dfres_stacked('sage')
dfg = rc.get_dfres_stacked()

utgcn = ut.buo(l_p)
# df_gcn = utgcn.get_results_dataframe_compiled()
# df_gcn = utgcn.get_full_results() # test = utgcn.gres
dfbig = utgcn.get_big_full()
# %%
# pd.options.display.float_format = '{:.2f}'.format
def mantisator(serie_):
    exponent = np.floor(np.log10(df_))
    mantissa = serie_/10**exponent
    mnt = mantissa.apply(lambda x: round(x, 2)) # original 
    mnts = mnt.apply(lambda x: f'{x:.2f}')
    exp0 = 'E' + exponent.astype(int).astype(str)
    return mnts + exp0
# df_ = dfbig.loc['Beam2D', 'std']
df_ = dfbig.loc['Beam2D', 'mean']
d = mantisator(df_)    
print(df_.T)
print(d)
dfc = dfbig.T.apply(lambda x: mantisator(x))
dfgT = dfbig.T
# %%
ltx = (dfc.T.to_latex(index=True,
                  formatters={"name": str.upper},
                  float_format="{:.2f}".format,))  
# %% tohle konecne funguje jak ma ... kde je chyba nevim
review = []
for _ in dfbig.index:
    print(_)
    df_ = dfbig.loc[_]
    d_ = mantisator(df_)    
    review.append(d_)
rdf = pd.concat(review, axis = 1)
# for _ in dfgT.columns:
#     # print(_)
#     d_ = mantisator(dfgT[_])
#     review.append(d_)
#     break
    # %%
rdf = pd.concat(review, axis = 1)# review dataframe 
# %%   test , pro se to tak sere 
dfgT = dfbig.T 
dfgT['Beam2D', 'mean']  = mantisator(dfgT['Beam2D', 'mean'])
# dfgT['Beam3D', 'mean']  = mantisator(dfgT['Beam3D', 'mean'])
