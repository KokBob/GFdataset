# -*- coding: utf-8 -*-
"""
https://realpython.com/python-zip-function/
"""
import sys
import os 
import pandas as pd
sys.path.append("..") 
from load_dataset import load_dataset 
from modelling.modelsStore import GCN0
from modelling.modelsStore import SAGE0
from modelling.preprocessing import *
import modelling.experimentationing as exps
import torch
import matplotlib.pyplot as plt
import glob as glob
import numpy as np
import pathlib
import numpy as np
import glob
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from utils_tab import buo
def drop_outlayer(_df):
    for col_ in _df.columns:
        mx = _df[col_].sort_values()
        mxv = mx.values
        mxv[-1] = mx[:-2].mean()
        # mxv[ 0] = mx[:-2].mean()
        _df[col_] = mxv
    _df = pd.DataFrame(_df)
    return _df
def get_time_df(dataset_ , framework_select = 'SAGE'):
    '''
    ''

    Parameters
    ----------
    dataset_ : TYPE: str: 'b2', 'b3', 'fs', 'pl'
        DESCRIPTION.
    framework_select : TYPE str: 'SAGE', 'GCN'
        DESCRIPTION. The default is 'SAGE' framework 

    Returns
    -------
    dfb : TYPE dataframe with experiments time based on pt* saving  
        for all type of reduction strategies 
    

    '''
    a, exp_set = [], ['Bench', 'ShP','TrS', 'L1', 'WL1', 'WL2']
    for _exp_label in exp_set:  
        path_test = f'../ffnet/{dataset_}/{_exp_label}'
        p_0 = r'./results_gred/'
        list_of_files_npy = glob.glob(f'{path_test}/*{framework_select}*.pt')
        list_of_files_npy.sort(key=os.path.getmtime)
        l = [pathlib.Path(file).lstat().st_mtime for file in list_of_files_npy]
        df = pd.DataFrame(l)
        df['date'] = [pd.Timestamp.fromtimestamp(x) for x in df[0]]
        df['datedif'] = df['date'].diff()
        df[_exp_label]= df['datedif'].astype('timedelta64[s]')
        a.append(df[_exp_label]) 
    dfa = pd.concat(a, axis = 1).apply(np.float64)
    dfa.iloc[0,:] = dfa.iloc[1,:] # derivace 0
    dfa = drop_outlayer(dfa)
    dfa = drop_outlayer(dfa)
    dfb = pd.DataFrame(dfa, columns = exp_set)
    # dfb['framework'] = framework_select.lower()
    return dfb
dataset_list = ['b2', 'b3', 'fs', 'pl']
list_dfs =[]
list_Js = []
for _ in dataset_list:
    # df_time_ = get_time_df(_, framework_select = 'GCN')
    df_time_ = get_time_df(_, framework_select = 'SAGE')
    list_dfs.append(df_time_)
    j = df_time_.describe()
    j = j.loc[['mean','std']]
    list_Js.append(j)
    name_dataset = [_]

df_ = pd.concat(list_Js)

def get_ltx_preped(full_dataframe):
    def mantisator(serie_, precision = 4):
        exponent = np.floor(np.log10(serie_))
        mantissa = serie_/10**exponent
        mnt = mantissa.apply(lambda x: round(x, precision)) 
        mnts = mnt.apply(lambda x: f'{x:.3f}') # nebo: f'{x:.2f}') etc
        exp0 = 'E' + exponent.astype(int).astype(str)
        return mnts + exp0
    df_results = full_dataframe
    df_compiled = df_results.apply(lambda x: mantisator(x)).T
    return df_compiled

ltx = get_ltx_preped(df_)
ltx_c = (ltx.T.to_latex(index=True,
                  formatters={"name": str.upper},
                  float_format="{:.2f}".format,))  

