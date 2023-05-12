import sys
import os 
sys.path.append("..") 
from evaluation import inspect_learning
#%%results loading
pathes = [r'./b2', r'./b3', r'./fs', r'./pl' ]

for path in pathes:
    insp = inspect_learning.Inspect()
    insp.results_compile(path)
    insp.plot_df_se()
    insp.plot_df_ne()
    insp.plot_df_mx()
    insp.plot_beasts()
    