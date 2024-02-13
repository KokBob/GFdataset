# -*- coding: utf-8 -*-
"""

"""
# from pathlib import Path
import numpy as np
import glob
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def get_dfres(df_, index_name):
    # index_name = 'gcn' || 'sage'
    c2 = np.vstack(np.array(df_.loc[index_name]))
    df2 = pd.DataFrame( c2.astype('float64').T, columns=df_.loc[index_name].index)
    df2['framework'] = index_name
    return df2 



p_0 = r'./results_gred/'
l_p = glob.glob(f'{p_0}*.npy')
file_ = l_p[1]
d = np.load(file_,allow_pickle=True).item()
df = pd.DataFrame.from_dict(d)
# %%
list_df_ = []
for _ in df.index: list_df_.append(get_dfres(df, index_name=_))
dfc = pd.concat(list_df_) # compilation df
# %%
a = [get_dfres(df, index_name=x) for x in df.index] 
dfc2 = pd.concat(a) # compilation df
# %%
# dfc.groupby('framework').boxplot()
# index_name = 'gcn' #|| 'sage'
# c1 = np.array(df.loc[index_name])
# c2 = np.vstack(np.array(df.loc[index_name]))
# c3 = np.hstack(np.array(df.loc[index_name]))
# df2 = pd.DataFrame( c2.astype('float64').T, columns=df.loc[index_name].index)
# df2['framework'] = index_name
# %%
# ids = list(df.loc[index_name].index)
# il = [ids]*c2.shape[1]
# ia = np.array(il).T # pro zajimavos jak michat : # ia = np.array(il)
# ias = np.hstack(ia)
# df3 = pd.DataFrame(c3.astype('float64'), columns=['acc'])
# df3['mt'] = ias
# df3['framework'] = index_name
# df3['dataset'] = 'Beam2D'
def get_dfres_stacked(df_, index_name):
    # stacked experiment by experiment
    # index_name = 'gcn' #|| 'sage'
    # df_ = df
    c3 = np.hstack(np.array(df_.loc[index_name]))
    ids = list(df_.loc[index_name].index)
    il = [ids]*df_.loc[index_name][0].shape[0]
    ia = np.array(il).T # pro zajimavos jak michat : # ia = np.array(il)
    ias = np.hstack(ia)
    
    
    df3 = pd.DataFrame(c3.astype('float64'), columns=['acc'])
    df3['mt'] = ias
    df3['framework'] = index_name

    return df3
# %%
# sns.factorplot("dataset", hue="framework", y="dataset", data=df, kind="box")
# df3.groupby('mt').boxplot()
# df3.boxplot()

# args = {'fname': 'wiu.png'}
# kwargs = {}
# plt.savefig(args)
dataset = file_.split('\\')[1].split('_')[0]
aa = [get_dfres_stacked(df, index_name=x) for x in df.index] 
dfc3 = pd.concat(aa) # compilation df

plt.figure()
'''
selector_y = 'steps' # sigma / eps
'''
ax = sns.boxplot(data=dfc3, x="mt", y='acc',
            # hue="mt", 
            hue="framework", 
            # notch=True, 
            showcaps=True,
            medianprops={"color": "k", "linewidth": 2},
            )
plt.yscale('log',)
plt.title(f'{dataset}')
plt.savefig(f'../graph_reduction/pics/{dataset}_hist_02.png')
