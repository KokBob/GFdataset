# -*- coding: utf-8 -*-

import networkx as nx
import pandas as pd
import dash
import dash_vtk
from dash import html
import numpy as np

print([x for x in dir(nx) if '_layout' in x])
# %%

# nodes = 20
# edges_num = 100
# H = nx.gnm_random_graph(nodes,edges_num,20)
# pos = nx.kamada_kawai_layout(H, dim = 3)
# %% Read  nodes
uzly = pd.read_csv('uzly.csv', names = ['Label', 'x', 'y', 'z'] )
# %% spojeni uzlu
elementi = pd.read_csv('elemts.csv', names = ['Label', 'n1', 'n2'] ) #n1 ... node 1, n2 node 2
elementi['nums'] = 2 # number of points at connection 
# %%
# https://networkx.org/documentation/stable/reference/generated/networkx.convert_matrix.from_numpy_matrix.html
# https://stackoverflow.com/questions/29572623/plot-networkx-graph-from-adjacency-matrix-in-csv-file
# https://stackoverflow.com/questions/56597840/drawing-weighted-graph-from-adjacency-matrix-with-edge-labels
# https://stackoverflow.com/questions/49683445/create-networkx-graph-from-csv-file
# Graphtype = nx.Graph()
# G = nx.from_pandas_edgelist(elementi,  create_using=Graphtype)
# G = nx.from_pandas_edgelist(elementi, edge_attr='weight', create_using=Graphtype)
# https://networkx.org/documentation/stable/tutorial.html
# %%
G = nx.Graph()
# uzly['Label']
G.add_nodes_from(uzly['Label'])
# %%
# https://stackoverflow.com/questions/9758450/pandas-convert-dataframe-to-array-of-tuples
# https://networkx.org/documentation/stable/reference/classes/generated/networkx.Graph.add_edge.html
# connect = elementi[['n1','n2']].values
# connect = list(elementi.itertuples(index=False))
connect =list(elementi[['n1','n2']].itertuples(index=False, name=None))
# G.add_edges_from([(1, 2), (1, 3)])
G.add_edges_from(connect)

# for _ in connect:
    # print(_)
    # u,v = _['n1'], _['n2']
    # print(str(u))
    # G.add_edge(*_)

print(G.edges)
# %%
xyz = uzly[['x','y','z']].values
# %%
# nb_points = 2
# lines_list = []
# for edge in G.edges:
#     l_ = (nb_points,) + edge
#     lines_list.append(l_)
# lines_01 = [item for sublist in lines_list for item in sublist]
# lines_01 =list(np.concatenate(connect))
# lines_01 = [0,1,2,3,4]
# lines_01 = np.concatenate(elementi.values)
# lines_01 = list(lines_01)
# https://dash.plotly.com/vtk/structure

lines_01 =list(elementi[['nums','n1','n2']].itertuples(index=False, name=None))

# %%
polys_01  = [2, 0, 1]
# %%
# points_01= [item for sublist in xyz for item in sublist]
# points_01= xyz
points_01=  list(np.concatenate(xyz))
cell_values=[1, 0]
point_values =  [item for sublist in np.random.rand(5,1).tolist() for item in sublist]
# %%

# Setup VTK rendering of PointCloud
app = dash.Dash(__name__)
server = app.server
app.layout = html.Div([dash_vtk.View([
    dash_vtk.GeometryRepresentation(
        children=[
            dash_vtk.PolyData(
                points =points_01,
                lines  = lines_01,
                # polys  = polys_01,
                children=[
                    
  
                    dash_vtk.PointData([
                        dash_vtk.DataArray(
                            registration='setScalars', # To activate field
                            name='onPoints',
                            values=point_values,
                            # pointSize= 2
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
    ])],
    
    style={"width": "100%", "height": "800px"
           }
    )


if __name__ == "__main__":    app.run_server(debug=True)
