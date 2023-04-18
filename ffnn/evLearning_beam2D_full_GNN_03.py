import sys
import os 
sys.path.append("..") 
from evaluation import inspect_learning
#%%results loading

# for 
path = r'./b2'
# path     = r'./fs'
insp = inspect_learning.Inspect()
insp.results_compile(path)


# insp.plot_df_se()
# insp.plot_df_ne()
# insp.plot_df_mx()
#%%results loading
df = insp.df_mx
# %%

import matplotlib.pyplot as plt


x = df.idxmin(axis=0)
y = df.min(axis=0)

# 
df.plot(logy=True,
          linewidth=0.5,
          color='grey',
          legend=None,
          # title = MeanSquareError'
          
          )
# plt.scatter(x,y, c=y)
plt.scatter(x,y, c=x)
plt.savefig('b2training.png')
plt.close()
# df.plot.scatter(x, y)
# df.plot.scatter(x='x', y='y')
