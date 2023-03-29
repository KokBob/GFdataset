import pandas as pd
from dash import Dash, dcc, html
import dash_bootstrap_components as dbc
import dash
import dash_vtk
from dash import html
import numpy as np
import vtk
from dash_vtk.utils import to_mesh_state
import plotly
import plotly.express as px
# plotly.offline.plot(vtk_view_y, filename='./vtk_y.html')
# https://kitware.github.io/vtk-examples/site/VTKBook/08Chapter8/
vtu_file = 'beam36.vtu'
vtk_file_path = vtu_file # without states


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
            colorDataRange=rng_x)
    ],
    background=[1, 1, 1],
    cameraPosition=[10,10,10], )

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
    cameraPosition=[10,10,10], 
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
    cameraPosition=[10,10,10], 
)

vtk_view_err = dash_vtk.View(
    [         dash_vtk.GeometryRepresentation(
            [ dash_vtk.Mesh(state = mesh_state_err), ],
            property={"edgeVisibility": True, "opacity": .9872},
            colorDataRange=rng,
             
        )
    ],
    background=[1, 1, 1],
    cameraPosition=[10,10,10], 
)


app = Dash(external_stylesheets=[dbc.themes.SLATE])    
# app.layout = html.Div( html.Div(vtk_view_x),)
app.layout = html.Div(
    style={"height": "calc(100vh - 16px)"},
    children=[
        html.H1(children='X'),
        html.Div(vtk_view_x, style={"height": "25%", "width": "50%", }),
        html.H1(children='Y'),
        html.Div(vtk_view_y, style={"height": "25%", "width": "50%", }),
        html.H1(children='Yhat'),
        html.Div(vtk_view_y, style={"height": "25%", "width": "50%", }),
        html.H1(children='error'),
        html.Div(vtk_view_y, style={"height": "25%", "width": "50%", }),
              ],
    
)

if __name__ == "__main__":    app.run_server(debug=True, port=8051)
