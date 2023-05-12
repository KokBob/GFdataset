import glob 
import numpy as np
import pandas as pd
#%%results loading
# path = r'./b2'
path            = r'./fs'

fish4           = path + '/**/*.npy'
list_of_files   = glob.glob(fish4, recursive=True)
numpy_file      = list_of_files[0]
R               = np.load(numpy_file, allow_pickle=True)
# df = pd.DataFrame(columns=['MSE', 'METHOD', 'IDS', 'T_end'], index=range(len(list_of_files)))
#%% gathering results
i = 0
c_  = R.item()

SE = []
NE = []
MX = []

for e_ in c_:
    # print(e_)
    name_ = ''.join(numpy_file.split('\\')[1].split('_')[1:3])
    se = c_[e_]['SumErr']
    ne = c_[e_]['NodeErr'].cpu().detach().numpy().item()
    mx = c_[e_]['loss_Rmax'].cpu().detach().numpy().item()
    SE.append(se)
    NE.append(ne)
    MX.append(mx)
    # break

# %%
# zipped = list(zip(SE,NE, MX ))
# df = pd.DataFrame(SE)
# df = pd.DataFrame(zipped, columns = ['SE', 'NE', 'rmax'])
# df.astype({'SE': 'float'}).dtypes
# df.astype({'NE': 'float'}).dtypes
# df = pd.DataFrame.from_dict(c_)
# %%rmax%%
# df['rmax'].plot()
# %%
# df0 = pd.DataFrame()
comp = {}
for numpy_file in list_of_files:
    SE = []
    R     = np.load(numpy_file, allow_pickle=True).item()
    name_ = ''.join(numpy_file.split('\\')[1].split('.')[0].split('_')[1:4])
    for e_ in R:
        # se = R[e_]['SumErr']
        se = R[e_]['loss_Rmax'].cpu().detach().numpy().item()
        SE.append(se)
    comp[name_] = SE
df0 = pd.DataFrame.from_dict(comp) 
    
# %%
df0.plot(logy=True,
         linewidth=0.5,
         color='grey',
         )

df0.max(axis=1).plot(logy=True,)
df0.mean(axis=1).plot(logy=True,)
df0.min(axis=1).plot(logy=True,)
# re_dict = c__.tolist()
# for  _ in list_of_files:
#     s_ = _.split('\\')[1]
#     t_ = _.split('\\')[2].split('_')[1]
#     r_ = np.load(_, allow_pickle=True).tolist()
#     mse_ = r_['losses_val'].min()
#     te_ = r_['time_elapsed'][-1]
#     df.iloc[i] = [mse_, t_, s_, te_ ]
#     i+=1
#     print(_)
# # %% savinf results to csv
# df.to_csv('violin_gf_13.csv')
