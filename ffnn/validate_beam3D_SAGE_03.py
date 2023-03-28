# -*- coding: utf-8 -*-
import sys
import pandas as pd

from sklearn.model_selection import train_test_split
sys.path.append("..") 
from load_dataset import load_dataset 
from modelling.modelsStore import SAGE0
from modelling.preprocessing import *
from modelling.experimentationing import *
import torch.nn.functional as F
import random 
from sklearn.preprocessing import MinMaxScaler
import torch
import time
from dgl.dataloading import GraphDataLoader

GFDS = load_dataset.Beam3D()
gfds_name = GFDS.gfds_name
pathRes  = GFDS.pathRes
methodID = 'SAGE_RF2' # FFNN,SAGE0,GCN0,
MODEL = SAGE0
D = GFDS.D
X0 = GFDS.X0
y = GFDS.y
G = GFDS.G
print(X0.max())
scaler = MinMaxScaler()
df = pd.DataFrame(X0)
X0 = pd.DataFrame(scaler.fit_transform(df), columns=df.columns).values
print(X0.max())
# %%
graphs = graphs_preparation(D, G, X0, y)
# %%
experiments_IDs = [42,17,23,11,18,4,5,1,6,1212]
loss_fn = F.mse_loss
num_epochs = 500
for experiment_number in experiments_IDs:
    
    experiment_name = f'{gfds_name}_{methodID}_{experiment_number}'
    random.seed(experiment_number)
    random.shuffle(graphs)

    split_number = int(len(graphs)*.7)
    train_loader = GraphDataLoader(graphs[:split_number],shuffle=True, )
    test_loader = GraphDataLoader(graphs[split_number:],shuffle=False, ) 
    
    for batch in train_loader: break    
    in_channels = batch.ndata['y'].shape[0]
    
    
    
    model = MODEL(in_channels, in_channels, in_channels)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    my_device = "cuda" if torch.cuda.is_available() else "cpu"    

    model = model.to(my_device)  
    
    pathModel = pathRes + experiment_name + '.pt'
    inputs = 'x'
    targets = 'y'
    break

# %%
model.load_state_dict(torch.load(pathModel))
model.eval()
# %%
for batch in test_loader:        
    batch = batch.to(my_device)
    x       = batch.ndata[inputs].cpu().numpy()
    y       = batch.ndata[targets].cpu().numpy()
    y_hat   = model(batch, batch.ndata[inputs]).cpu().detach().numpy()
    err     = y - y_hat
    # plt.scatter(y_hat, err,  c=err, alpha=0.5)
    # plt.scatter(y, y_hat,c=err, alpha=0.5)
    # plt.scatter(x, err,c=err, alpha=0.5)
    # plt.scatter(x, y_hat, c=err, alpha=0.5)
    # plt.hist(err, bins = 120) 
    # plt.show()
    # break
# %% vtk ground truth 
# -*- coding: utf-8 -*-
'''
'''
import pandas as pd
import dash
import dash_vtk
from dash import html
import numpy as np
# https://kitware.github.io/vtk-examples/site/VTKBook/08Chapter8/
vtu_file = 'beam36.vtu'
# vtu_file = 'fibonacci3.vtu'
# vtu_file = 'plane3.vtu'
# nodes = pd.read_csv('nodes.csv')
nodes = pd.read_csv('nodes_Beam3D.csv')
# nodes = pd.read_csv('nodes_fibonacci.csv')
# nodes = pd.read_csv('nodes_plane.csv')
# nodes = pd.read_csv('nodes_b7.csv')
# nodes = pd.read_csv('nodes_tq.csv')
nodes = nodes[['X', 'Y', 'Z']]
xyz = list(nodes.values)
points_01= [item for sublist in xyz for item in sublist]
# point_values =  [item for sublist in np.random.rand(len(xyz),1).tolist() for item in sublist]
point_values =  x
# point_values =  y
# point_values =  y_hat
# point_values =  err

# elements = pd.read_csv('elements.csv')
elements = pd.read_csv('elements_Beam3D.csv')
# elements = pd.read_csv('elements_fibonacci.csv')
# elements = pd.read_csv('elements_plane.csv')
# elements = pd.read_csv('elements_b7.csv')
# elements = pd.read_csv('elements_tq.csv')
elements = elements.loc[:, elements.columns != 'ID']
elements = elements -min(elements.min().values)
elements['count']=elements.shape[1]
# elements['type']=24 # quatratic tetra
elements['type']=12 # linear hexa
# elements['type']=10 # linear tetra
df_offset = elements['count']
del elements['count']
df_offset = df_offset.cumsum()
df_type = elements['type']
del elements['type']
# elements['count']=10
# %%
l1 = '<VTKFile type="UnstructuredGrid" version="0.1" byte_order="LittleEndian"> \n'
l2 = '<UnstructuredGrid> \n'
l3 = '<Piece NumberOfPoints="' + str(len(nodes)) + '" NumberOfCells="'+ str(len(elements)) +'"> \n'
l4 = '<Points> \n'
l5 = '<DataArray type="Float64" NumberOfComponents="3" format="ascii"> \n'
l6 =  nodes.to_string(header=False, index=False) 
l7 = '\n</DataArray> \n'
l8 = '</Points> \n'
l81 = '<PointData Tensors="" Vectors="" Scalars="">'
l82 = '\n<DataArray type="Float32" Name="X_" format="ascii"> \n'
# l83 = pd.DataFrame(point_values).to_string(header=False, index=False)
l83 = pd.DataFrame(x).to_string(header=False, index=False)
l84 = '\n</DataArray> \n'

l82b = '\n<DataArray type="Float32" Name="Y_" format="ascii"> \n'
l83b = pd.DataFrame(y).to_string(header=False, index=False)
l84b = '\n</DataArray> \n'

l82c = '\n<DataArray type="Float32" Name="Yhat_" format="ascii"> \n'
l83c = pd.DataFrame(y_hat).to_string(header=False, index=False)
l84c = '\n</DataArray> \n'

l82d = '\n <DataArray type="Float32" Name="err_" format="ascii"> \n'
l83d = pd.DataFrame(err).to_string(header=False, index=False)
l84d = '\n</DataArray> \n'

l85 = '</PointData> \n' 
l9 = '<Cells> \n'
l10 = '<DataArray type="Int32" Name="connectivity" format="ascii"> \n'
l11 =  elements.to_string(header=False, index=False) 
l11a = '\n</DataArray>\n'
l12 = '\n<DataArray type="Int32" Name="offsets" format="ascii">\n'
l13 = df_offset.to_string(header=False, index=False)
l14 = '\n</DataArray>\n'
l15 = '<DataArray type="UInt8" Name="types" format="ascii">\n'
l16 = df_type.to_string(header=False, index=False) 
l17 ='\n</DataArray>\n'
l18 ='</Cells>\n'
l19 ='</Piece>\n'
l20 ='</UnstructuredGrid>\n'
l21 ='</VTKFile>\n'

lines = [l1, l2,l3, l4, l5, l6, l7, l8,
         l81,
         l82,l83,l84,         
         l82b,l83b,l84b,
         l82c,l83c,l84c,
         l82d,l83d,l84d,
         l85,           
         l9, l10, l11,l11a,
         l12, l13, l14, l15, l16,l17,l18,l19,l20,l21]
with open(vtu_file, 'w') as f:
    
    f.writelines(lines)
    
vtk_file_path = vtu_file # without states

import vtk
from dash_vtk.utils import to_mesh_state
reader = vtk.vtkXMLUnstructuredGridReader()
reader.SetFileName(vtk_file_path)
reader.Update() 
# mesh_state =to_mesh_state(reader.GetOutput(), )
mesh_state_X =to_mesh_state(reader.GetOutput(), field_to_keep= 'X_')
mesh_state_Y =to_mesh_state(reader.GetOutput(), field_to_keep= 'Y_')
mesh_state_Yhat =to_mesh_state(reader.GetOutput(), field_to_keep= 'Yhat_')
mesh_state_err =to_mesh_state(reader.GetOutput(), field_to_keep= 'err_')
rng_x = [0, 1]
rng = [0, 10]
# cs_name = 'Greens'
# cs_name = 'Greens'

app = dash.Dash(__name__)
server = app.server
vtk_view_x = dash_vtk.View(
    [ dash_vtk.GeometryRepresentation(
        [dash_vtk.Mesh(state = mesh_state_X), ],
            property={"edgeVisibility": True, "opacity": .9872},
            # colorMapPreset=cs_name,
            colorDataRange=rng_x
        )
    ],
    background=[1, 1, 1],
)

vtk_view_y = dash_vtk.View(
    [
         dash_vtk.GeometryRepresentation(
            [
                dash_vtk.Mesh(state = mesh_state_Y),       
                
             ],

            property={"edgeVisibility": True, "opacity": .9872},
            # colorMapPreset=cs_name,
            colorDataRange=rng
        )
    ],
    background=[1, 1, 1],
)

vtk_view_yhat = dash_vtk.View(
    [
         dash_vtk.GeometryRepresentation(
            [
                dash_vtk.Mesh(state = mesh_state_Yhat),       
                   
             ],

            property={"edgeVisibility": True, "opacity": .9872},
            # colorMapPreset=cs_name,
            colorDataRange=rng
        )
    ],
    background=[1, 1, 1],
)

vtk_view_err = dash_vtk.View(
    [         dash_vtk.GeometryRepresentation(
            [ dash_vtk.Mesh(state = mesh_state_err), ],
            property={"edgeVisibility": True, "opacity": .9872},
            # colorMapPreset=cs_name,
            colorDataRange=rng
        )
    ],
    background=[1, 1, 1],
)

app.layout = html.Div(
    style={"height": "calc(100vh - 16px)"},
    children=[
        html.H1(children='X'),
        html.Div([vtk_view_x,vtk_view_y], style={"height": "15%", "width": "50%", }),
        html.H1(children='Y'),
        html.Div(vtk_view_y, style={"height": "15%", "width": "50%", }, ),
        html.H1(children='Yhat'),
        html.Div(vtk_view_yhat, style={"height": "15%", "width": "50%", }),
        html.H1(children='error'),
        html.Div(vtk_view_err, style={"height": "15%", "width": "50%", }),
              ],
    
)

if __name__ == "__main__":    app.run_server(debug=True)

