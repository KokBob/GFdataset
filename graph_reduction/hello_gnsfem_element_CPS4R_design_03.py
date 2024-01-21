'''
https://www.datacamp.com/community/tutorials/networkx-python-graph-tutorial?utm_source=adwords_ppc&utm_campaignid=898687156&utm_adgroupid=48947256715&utm_device=c&utm_keyword=&utm_matchtype=b&utm_network=s&utm_adpostion=&utm_creative=229765585183&utm_targetid=dsa-429603003980&utm_loc_interest_ms=&utm_loc_physical_ms=1000824&gclid=EAIaIQobChMIq9-J94a26wIVD9myCh3OJA1BEAAYASAAEgJK6PD_BwE
https://www.researchgate.net/post/How_do_I_run_a_python_code_in_the_GPU
https://developer.nvidia.com/blog/gpu-accelerated-graph-analytics-python-numba/
conda install numba cudatoolkit
Finding dense subgraphs via low-rank bilinear optimization
https://dl.acm.org/doi/10.5555/3044805.3045103
gpu netowrkx
https://towardsdatascience.com/4-graph-algorithms-on-steroids-for-data-scientists-with-cugraph-43d784de8d0e
Nvidia Rapids cuGraph: Making graph analysis ubiquitous
https://www.zdnet.com/article/nvidia-rapids-cugraph-making-graph-analysis-ubiquitous/
https://anaconda.org/rapidsai/cugraph
https://www.geeksforgeeks.org/running-python-script-on-gpu/
conda install numba & conda install cudatoolkit

https://stackoverflow.com/questions/42558165/load-nodes-with-attributes-and-edges-from-dataframe-to-networkx
'''
#import dgl
import networkx as nx
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import gnsfem as gf
from gnsfem import preprocessing
from gnsfem import visualization
#%% select tes

file_name = r'beam-01.inp'  # 2D mesh
# file_name = r"Fibo-1.inp"   #fibo
#file_name = r"Job-2.inp"   #fine meshed beam
#file_name   = r"Job-3.inp"    #coarse meshed beam
#file_name   = r'Job-ovalization-test-file.inp'
#%%
file1       = open(file_name,"r")
d           = file1.readlines() 
#%% get lines number ... old method
line_nodes, line_elements, line_end = preprocessing.get_lines_number(d, endDef = '*Nset, nset=Set-Part, generate')
# %% New method to extract informations from inp file 
linesDict = preprocessing.getElementAndNodeLines(d)
# %% old method
# d_nodes         = preprocessing.get_nodes_2D(file_name, line_elements,line_nodes)
# d_elements      = preprocessing.get_edges(file_name, line_elements,line_nodes, line_end)
# %% new method ... 
# tohle se da jeste zelegantnit
# line_elements,line_nodes, line_end = linesDict['*Node'][1], linesDict['*Node'][0], linesDict['*Element'][0]
d_nodes         = preprocessing.get_nodes_2D(file_name, linesDict['*Node'][0],linesDict['*Node'][1])
d_elements      = preprocessing.get_edges(file_name, line_elements,line_nodes, line_end)
#%% test to see nodes
# visualization.test_see_nodes_2D(d_nodes)
#%% create graph connections and how to add coordinates for graph
G = preprocessing.create_graph_coor_2D(d_nodes, d_elements)
# Gc = preprocessing.create_graph_coor_dgl_2D(d_nodes, d_elements) #AttributeError: 'DGLHeteroGraph' object has no attribute 'add_node'
G.name = 'Beam2D'
# nx.write_adjlist(G, G.name + ".adjlist")
#%% create position ... 
pos= d_nodes[['x','y','z']].values
# pos=nx.get_node_attributes(Gc,'pos')
#%%
nx.draw(G, with_labels = True, font_size = 15) # with no coordinates
# nx.draw(Gc, pos = pos, with_labels = True)      # with  coordinates
# %%
# https://docs.dgl.ai/en/0.4.x/generated/dgl.DGLGraph.from_networkx.html
# import dgl
# g = dgl.DGLGraph()
# gc = dgl.from_networkx(Gc)

