# -*- coding: utf-8 -*-
"""
"""
import glob 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
class Inspect(object):
    def __init__(self,): pass
    def results_compile(self, path = r'./fs'):
        comp_se = {}
        comp_ne = {}
        comp_mx = {}
        self.id = path.split('./')[1]
        # path            = r'./fs'
        fish4           = path + '/**/*.npy'
        list_of_files   = glob.glob(fish4, recursive=True)
        for numpy_file in list_of_files:
            SE, NE, MX = [],[],[]
            R=np.load(numpy_file, allow_pickle=True).item()
            name_ = ''.join(numpy_file.split('\\')[1].split('.')[0].split('_')[1:4])
            for e_ in R:
                se = R[e_]['SumErr']
                ne = R[e_]['NodeErr'].cpu().detach().numpy().item()
                mx = R[e_]['loss_Rmax'].cpu().detach().numpy().item()
                SE.append(se)
                NE.append(ne)
                MX.append(mx)
            comp_se[name_] = SE
            comp_ne[name_] = NE
            comp_mx[name_] = MX
            
        self.df_se = pd.DataFrame.from_dict(comp_se) 
        self.df_ne = pd.DataFrame.from_dict(comp_ne) 
        self.df_mx = pd.DataFrame.from_dict(comp_mx) 
        
    def plot_df_se(self):
        df__ = self.df_se
        df__.plot(logy=True,
                  linewidth=0.5,
                  color='grey',
                  legend=None,
                  title = self.id + 'RootMeanSquareError'
                  
                  )

        df__.max(axis=1).plot(logy=True,)
        df__.mean(axis=1).plot(logy=True,)
        df__.min(axis=1).plot(logy=True,)
        plt.savefig(self.id + 'RootMeanSquareError.png')
        # df__.plot(legend=None)
        # plt.title('SumError')
    def plot_df_ne(self):

        df__ = self.df_ne
        df__.plot(logy=True,
                  linewidth=0.5,
                  color='grey',
                  legend=None,
                  title = self.id + 'NodeError'
                  )

        df__.max(axis=1).plot(logy=True,)
        df__.mean(axis=1).plot(logy=True,)
        df__.min(axis=1).plot(logy=True,)
        plt.savefig( self.id + 'NodeError.png')
        # plt.legend(bbox_to_anchor=(1,1), loc="upper left")
        # df__.plot(legend=None)
        # plt.title('nodeError')
    def plot_df_mx(self):
        df__ = self.df_mx
        df__.plot(logy=True,
                  linewidth=0.5,
                  color='grey',
                  legend=None,
                  title = self.id + 'RootMaxSquareError'
                  )

        df__.max(axis=1).plot(logy=True,)
        df__.mean(axis=1).plot(logy=True,)
        df__.min(axis=1).plot(logy=True,)
        plt.savefig(self.id + 'RootMaxSquareError.png')
        # df__.plot(legend=None)
    # def plot_df(self, df__):
    #     df__.plot(logy=True,
    #              linewidth=0.5,
    #              color='grey',
    #              )

    #     df__.max(axis=1).plot(logy=True,)
    #     df__.mean(axis=1).plot(logy=True,)
    #     df__.min(axis=1).plot(logy=True,)
        # for e_ in c_:
        #     # print(e_)
        #     se = c_[e_]['SumErr']
        #     ne = c_[e_]['NodeErr'].cpu().detach().numpy().item()
        #     mx = c_[e_]['loss_Rmax'].cpu().detach().numpy().item()
        #     SE.append(se)
        #     NE.append(ne)
        #     MX.append(mx)
        # zipped = list(zip(SE,NE, MX ))
        # # df = pd.DataFrame(SE)
        # df = pd.DataFrame(zipped, columns = ['SE', 'NE', 'rmax'])
        # # df.astype({'SE': 'float'}).dtypes
        # # df.astype({'NE': 'float'}).dtypes
        # # df = pd.DataFrame.from_dict(c_)
        # df.plot()
