# -*- coding: utf-8 -*-
"""

"""
# from pathlib import Path
import numpy as np
import glob
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def get_dfres_all(df_, index_name, file_):
    # index_name = 'gcn' || 'sage'
    c2 = np.vstack(np.array(df.loc[index_name]))
    df2 = pd.DataFrame( c2.astype('float64').T, columns=df.loc[index_name].index)
    # df2 = pd.DataFrame( c2.astype('float64'), columns=df.loc[index_name].index) # test strileni od boku
    df2['framework'] = index_name
    df2['dataset'] =file_.split('\\')[1].split('_')[0]
    df2 = df2.T # test
    return df2 

p_0 = r'./results_gred/'
l_p = glob.glob(f'{p_0}*.npy')
_all = []
for _file_ in l_p:
    d = np.load(_file_,allow_pickle=True).item()
    df = pd.DataFrame.from_dict(d)
    dfc = pd.concat([get_dfres_all(df, index_name=x, file_=_file_) for x in df.index] )
    _all.append(dfc)

dfall = pd.concat(_all)
df = dfall
# df = df.T
df.reset_index()

# %%
# dfg = dfall.groupby('framework')
dfg = dfall.groupby(['framework', 'dataset'])
dfg_description = dfg.describe()
dfg = dfg.describe()
# %%

ds = sns.load_dataset('tips')
# sns.factorplot("dataset", hue="framework", y="dataset", data=df, kind="box")
# sns.catplot(kind='box', data=df,)
# %%
ax = sns.boxplot(data=df, 
                  x=df.columns[0:-2], 
                  # y='dataset',
                  # hue="datasets", 
                 notch=True, showcaps=False,
                 medianprops={"color": "r", "linewidth": 2},
                    )
# %% overall good results 
dfg['Bench'] = dfg['Bench'][['mean','std']] 
# %%
# pd.reshape()
df1 = df.reset_index()
# df1 = df.rename(columns={'A':'A1', 'B':'B1', 'A1':'A2', 'B1':'B2'}).reset_index()
# pd.wide_to_long(df1, stubnames=['A', 'B'], i='index', j='id').reset_index()[['A', 'B', 'id']]
# %%
# df2 = pd.DataFrame( c2.astype('float64').T, columns=df.loc[index_name].index)

# pivoted = df.pivot(index=df1.index, columns="framework", values="dataset")
pivoted = df.pivot(columns="framework", values="dataset")
