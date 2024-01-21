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
from dash import Dash, dcc, html, Input, Output, no_update
import dash
import dash_vtk
#%% select test experiment
# file_name   = r'plane_simple.inp'
# file_name   = r'D:/Abaqus/05_Obeam/Obeam.inp'
file_name   = r'D:/Abaqus/05_Obeam/tetra.inp'
#%%
file1       = open(file_name,"r")
d           = file1.readlines() 
#%% get lines number
line_nodes, line_elements, line_end = preprocessing.get_lines_number(d)
d_nodes         = preprocessing.get_nodes(file_name, line_elements,line_nodes)
d_elements      = preprocessing.get_edges(file_name, line_elements,line_nodes, line_end)
#%% test to see nodes
# visualization.test_see_nodes_2D(d_nodes)
# visualization.test_see_nodes(d_nodes)
#%% create graph connections and how to add coordinates for graph

def create_graph_coor_tetra(d_nodes, d_elements):
    G = nx.Graph()
    for nd in d_nodes.index: 
        src0 = d_nodes.iloc[nd]
        src = int(src0['#Node'])
        pos0 = (src0['x'], src0['y'], src0['z'])
        G.add_node(src, pos = pos0)
    for i in range(np.shape(d_elements.index)[0]):
        els = d_elements.iloc[i].values 
        
        ea = els[0:]
        e2 = ea.reshape([2,2]).T
        G.add_edges_from(e2)
        e3 = ea.reshape([2,2])
        G.add_edges_from(e3)
        e4 =np.roll(e3,1)
        G.add_edges_from(e4)
    return G
G = preprocessing.create_graph_coor_tetra(d_nodes, d_elements)
# %
# %%
# nx.draw(G, with_labels = True)
# nx.draw(G)
# %%
# visualization.dashplotly_graph_initial_visualization(d_nodes, G )
# %%
# -*- coding: utf-8 -*-
uzly = d_nodes
xyz = uzly[['x','y','z']].values 

elementi  = pd.DataFrame(G.edges)
# %%
elementi['nums'] = 2
dummyEl = elementi.loc [:, elementi.columns != 'nums'] 
elementi.loc [:, elementi.columns != 'nums'] = elementi.loc [:, elementi.columns != 'nums'] - np.ones(dummyEl.shape)
elementi.set_index(elementi.pop('nums'), inplace=True)
elementi.reset_index(inplace=True)

# %%
lines_list =list(elementi.itertuples(index=False, name=None))
# lines_list =list(elementi[['nums','n1','n2']].itertuples(index=False, name=None))
lines_01 = [item for sublist in lines_list for item in sublist]

polys_01  = [2, 0, 1]

points_01=  list(np.concatenate(xyz))
cell_values=[1, 0]
point_values =  [item for sublist in np.random.rand( len(points_01) ,1).tolist() for item in sublist]
# %%
app = dash.Dash(__name__)
server = app.server
app.layout = html.Div(
    
    [dash_vtk.View(
        [
            dash_vtk.GeometryRepresentation(
                children=[
                    dash_vtk.PolyData(
                        points =points_01,
                        lines  = lines_01,
                        # polys  = polys_01,
                        children=[
                            
          
                            dash_vtk.PointData([
                                dash_vtk.DataArray(
                                    registration='setScalars', 
                                    name='onPoints',
                                    values=point_values,
                                )
                            ]),
                            
                            dash_vtk.CellData([
                                dash_vtk.DataArray(
                                    registration='setScalars', # To activate field
                                    name='onCells',
                                    values=cell_values,)
                            ]),
                                               
                            dash_vtk.PointCloudRepresentation(
                                xyz=points_01,
                                scalars=point_values,
                                # colorDataRange=[min_elevation, max_elevation],
                                property={"pointSize": 7, 
                                          "symbol": 'circle', },
                            ),  
                            
                        ],
                    ),
                ],
        
            ),
        
        ],
        background=[1, 1, 1],    
    )
    ],
    
    style={"width": "100%", 
            "height": "800px",
            # "height":"100%"

            }
    ,
    )


if __name__ == "__main__":    app.run_server(debug=True)
# 
