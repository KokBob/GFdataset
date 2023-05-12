import sys
import os 
sys.path.append("..") 
from evaluation import inspect_learning
#%%results loading
path = r'./b2'
# path     = r'./fs'
insp = inspect_learning.Inspect()
insp.results_compile(path)
insp.plot_df_se()
insp.plot_df_ne()
insp.plot_df_mx()
#%%results loading
path = r'./b3'
# path     = r'./fs'
insp = inspect_learning.Inspect()
insp.results_compile(path)

insp.plot_df_se()
insp.plot_df_ne()
insp.plot_df_mx()
#%%results loading
path = r'./fs'
# path     = r'./fs'
insp = inspect_learning.Inspect()
insp.results_compile(path)

insp.plot_df_se()
insp.plot_df_ne()
insp.plot_df_mx()
#%%results loading
path = r'./pl'
# path     = r'./fs'
insp = inspect_learning.Inspect()
insp.results_compile(path)

insp.plot_df_se()
insp.plot_df_ne()
insp.plot_df_mx()
# %%
# import matplotlib.pyplot as plt
# plt.savefig('RootMeanSquareError.png')