# -*- coding: utf-8 -*-

import networkx as nx
import pandas as pd
import dash
import dash_vtk
from dash import html
import numpy as np

# uzly
uzly = pd.read_csv('uzly.csv', names = ['Label', 'x', 'y', 'z'] )
# spojeni uzlu
elementi = pd.read_csv('elemts.csv', names = ['Label', 'n1', 'n2'] ) #n1 ... node 1, n2 node 2
elementi['nums'] = 2 # number of points at connection 
elementi[['n1','n2']] = elementi[['n1','n2']] -1

G = nx.Graph()
G.add_nodes_from(uzly['Label'])
connect =list(elementi[['n1','n2']].itertuples(index=False, name=None))
G.add_edges_from(connect)
# %% 
xyz = uzly[['x','y','z']].values

lines_list =list(elementi[['nums','n1','n2']].itertuples(index=False, name=None))
lines_01 = [item for sublist in lines_list for item in sublist]

polys_01  = [2, 0, 1]

points_01=  list(np.concatenate(xyz))
cell_values=[1, 0]
point_values =  [item for sublist in np.random.rand(5,1).tolist() for item in sublist]
# %%
app = dash.Dash(__name__)
server = app.server
app.layout = html.Div([
    dash_vtk.View(
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
        # background=[1, 1, 1],    
    )
    ],
    
    style={"width": "100%", 
            "height": "800px",
           # "height":"100%"

           }
    ,
    )


if __name__ == "__main__":    app.run_server(debug=True)
