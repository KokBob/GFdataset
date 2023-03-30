import pandas as pd
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
def drawVTK(mesh_state_):
    vtk_view_ = dash_vtk.View(
        [dash_vtk.GeometryRepresentation([dash_vtk.Mesh(state = mesh_state_),],
                property={"edgeVisibility": True, "opacity": .9872},
                # colorMapPreset=cs_name,
                colorDataRange=rng
            )
        ],
        background=[1, 1, 1],
        cameraPosition=[10,10,10],)
    # return html.Div(vtk_view_, style={"height": "50%", "width": "50%", }) 
    return  html.Div([
        dbc.Card(
            dbc.CardBody([
                dcc.Graph(vtk_view_
                ) 
            ])
        ),  
    ])
from dash import Dash, dcc, html
import dash_bootstrap_components as dbc
import plotly.express as px

# Iris bar figure
def drawFigure():
    return  html.Div([
        dbc.Card(
            dbc.CardBody([
                dcc.Graph(
                    figure=px.bar(
                        df, x="sepal_width", y="sepal_length", color="species"
                    ).update_layout(
                        template='plotly_dark',
                        plot_bgcolor= 'rgba(0, 0, 0, 0)',
                        paper_bgcolor= 'rgba(0, 0, 0, 0)',
                    ),
                    config={
                        'displayModeBar': False
                    }
                ) 
            ])
        ),  
    ])

# Text field
def drawText():
    return html.Div([
        dbc.Card(
            dbc.CardBody([
                html.Div([
                    html.H2("Text"),
                ], style={'textAlign': 'center'}) 
            ])
        ),
    ])

# Data
df = px.data.iris()

# Build App
app = Dash(external_stylesheets=[dbc.themes.SLATE])

app.layout = html.Div([
    dbc.Card(
        dbc.CardBody([
            dbc.Row([
                dbc.Col([
                    drawText()
                ], width=3),
                dbc.Col([
                    drawText()
                ], width=3),
                dbc.Col([
                    drawText()
                ], width=3),
                dbc.Col([
                    drawText()
                ], width=3),
            ], align='center'), 
            html.Br(),
            dbc.Row([
                dbc.Col([
                    drawVTK(mesh_state_X )
                ], width=3),
                dbc.Col([
                    drawVTK(mesh_state_X )
                ], width=3),
                dbc.Col([
                    drawVTK(mesh_state_X )
                ], width=6),
            ], align='center'), 
            html.Br(),
            dbc.Row([
                dbc.Col([
                    drawFigure()
                ], width=9),
                dbc.Col([
                    drawFigure()
                ], width=3),
            ], align='center'),      
        ]), color = 'dark'
    )
])

# Run app and display result inline in the notebook
app.run_server()

if __name__ == "__main__":    app.run_server(debug=True)

