# -*- coding: utf-8 -*-
"""
https://saturncloud.io/blog/how-to-set-decimal-precision-of-a-pandas-dataframe-column-with-decimal-datatype/
"""
import pandas as pd
import numpy as np
from graph_reduction import resco
from decimal import Decimal, ROUND_HALF_UP

class buo:
    def __init__(self,list_of_results_file,
                 framework_method_ = 'gcn'):
        self.l_p = list_of_results_file
        self.framework_method = framework_method_
    def get_results_dataframe_compiled(self, ):
        '''
        Parameters
        ----------
        list_of_results_file : list of numpy compiled results
            DESCRIPTION.
        framework_method_ : str
            DESCRIPTION. The default is 'gcn', next: 'sage'.
    
        Returns
        -------
        dataframe of all methods compiled results.
        Example
        -------   
        # dfr_gcn = ut.get_results_dataframe_compiled(l_p, )   
        # dfr_sage = ut.get_results_dataframe_compiled(l_p, 'sage')
        # df_overview = dfr_gcn - dfr_sage
    
        '''
        ares, pares = [], []
        for _ in self.l_p:
            file_ = _
            rc_ = resco(_)
            dfs = rc_.get_dfres_stacked(self.framework_method)
            ar = rc_.to_res_array()
            ares.append(ar)
            pares.append(rc_.df_res)
        self.ares = ares 
        return pd.DataFrame(ares)
    def get_full_results(self):
        df_res = self.get_results_dataframe_compiled()
        gres = []
        names = []
        for _ in self.l_p:
            rc_ = resco(_)
            dfs = rc_.get_dfres_stacked(self.framework_method)
            ar = rc_.to_res_array()
            # ares.append(ar)
            # pares.append(rc_.df_res)
            g_ = dfs.groupby('mt')
            g_d = g_.describe()
            # percentiles=None
            df_acc = g_d['acc'][['mean','std']]
            gres.append(df_acc)
            ds_name = rc_.dataset_name
            names.append(ds_name)
        names = [val for val in names for _ in (0, 1)]
        self.gres = gres
        self.names = names
        return pd.concat(gres,axis=1)
    def get_big_full(self,):
        df_fres = self.get_full_results()
        al = [self.names,['mean', 'std']*4] # 4 datasety
        tuples = list(zip(*al))
        index = pd.MultiIndex.from_tuples(tuples)
        df_fres.columns = pd.MultiIndex.from_tuples(tuples, names=['Caps','Lower'])
        df_fres_T = df_fres.T
        return df_fres_T
    def mantisator(self,serie_, precision = 4):
        exponent = np.floor(np.log10(serie_))
        mantissa = serie_/10**exponent
        mnt = mantissa.apply(lambda x: round(x, precision)) 
        mnts = mnt.apply(lambda x: f'{x:.3f}') # nebo: f'{x:.2f}') etc
        exp0 = 'E' + exponent.astype(int).astype(str)
        return mnts + exp0
    def get_ltx_preped(self,):
        df_results = self.get_big_full()
        df_compiled = df_results.apply(lambda x: self.mantisator(x)).T
        return df_compiled
