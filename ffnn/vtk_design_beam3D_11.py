import pandas as pd
import plotly.graph_objects as go # or plotly.express as px
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
import plotly.io as pio
# plotly.offline.plot(vtk_view_y, filename='./vtk_y.html')
# https://kitware.github.io/vtk-examples/site/VTKBook/08Chapter8/
# https://discourse.vtk.org/t/vtkimagedata-to-png/8418/3
# https://dash-bootstrap-components.opensource.faculty.ai/docs/themes/
# ref[0]> https://towardsdatascience.com/3d-mesh-models-in-the-browser-using-python-dash-vtk-e15cbf36a132
# ref[0]> https://bootswatch.com/
# https://www.mecharithm.com/latex-to-html/
# https://github.com/facultyai/dash-bootstrap-components/issues/286

# vtk_file_path = 'beam36.vtu'
# vtk_file_path = 'beam3D1.vtu'
vtk_file_path = 'beam3D2.vtu'
reader = vtk.vtkXMLUnstructuredGridReader()
reader.SetFileName(vtk_file_path)
reader.Update() 
mesh_state      = to_mesh_state(reader.GetOutput(), )
mesh_state_X    = to_mesh_state(reader.GetOutput(), field_to_keep= 'X_')
mesh_state_y    = to_mesh_state(reader.GetOutput(), field_to_keep= 'Y_')
mesh_state_yhat = to_mesh_state(reader.GetOutput(), field_to_keep= 'Yhat_')
mesh_state_err  = to_mesh_state(reader.GetOutput(), field_to_keep= 'err_')
rng_x   = [0, 1]
rng     = [0, 10]
# cs_name = 'Greens'
# cs_name = 'Greens'

app = dash.Dash(__name__)
server = app.server


def generateVTKviewHTML(MESH_STATE, RANGE_SET = None):
    # cs_name = 'Greens'
    cs_name = 'Plasma (matplotlib)'
    # https://github.com/Kitware/vtk-js/blob/master/Sources/Rendering/Core/ColorTransferFunction/ColorMaps.json
    if not RANGE_SET:   rng     = [0, 10]
    else:               rng     = RANGE_SET
    vtk_view = dash_vtk.View(
        
                [dash_vtk.GeometryRepresentation(
                    [ dash_vtk.Mesh(state = MESH_STATE), ],
                        property={"edgeVisibility": True, 
                                  "opacity": .9872,
                                  # "showScalarBar":  True, # funguje yto NE
                                  }, 
                        colorDataRange=rng,
                        colorMapPreset=cs_name,
                        # showScalarBar=  True # funguje yto NE
                        )
                    ],
    pickingModes=["hover"],
    background=[1, 1, 1],
    cameraPosition=[10,25,35], 
    )
    
    return html.Div(
        style={"height": "calc(100vh - 16px)"},
        children=[html.Div(vtk_view, style={"height": "80%", "width": "100%", })]
        )
    

def create_bar(range_def = [-10,10] ):
    # y = [-10,10]
    y = range_def
    min_x = -0.015
    max_x = 0.015
    min_y = -0.3
    max_y = 0.3
    
    dummy_trace=go.Scatter(x=[min_x, max_x],
                 y=[min_y, max_y],
                 mode='markers',
                 marker=dict(
                     size=(max(y)-min(y))/100, 
                     color=[min(y), max(y)], 
                      colorscale='plasma',  
                     # colorscale='Greens',  
                     
                      # colorbar=dict(thickness=100), 
                      colorbar=dict(thickness=100, orientation='h'), 
                     showscale=True
                 ),
                 hoverinfo='none',
                 # plot_bgcolor='rgba(0,0,0,0)'
                )
    
    layout = dict(xaxis=dict(visible=False), yaxis=dict(visible=False), 
                  plot_bgcolor='rgba(0,0,0,0)'
                  )
    fig_bar = go.Figure([dummy_trace], layout)
    fig_bar = [dcc.Graph(figure=fig_bar)]
    return fig_bar

fig_bara = create_bar()
# app = dash.Dash()
# app.layout = html.Div([
    # dcc.Graph(figure=fig_bara)
# ])
vtk_view_0_html      =   generateVTKviewHTML(mesh_state, RANGE_SET = None)    
vtk_view_x_html      =   generateVTKviewHTML(mesh_state_X, RANGE_SET = [0,1])
vtk_view_y_html      =   generateVTKviewHTML(mesh_state_y, RANGE_SET = None)
vtk_view_yhat_html   =   generateVTKviewHTML(mesh_state_yhat, RANGE_SET = None)
vtk_view_err_html    =   generateVTKviewHTML(mesh_state_err, RANGE_SET = [-0.005,.02])


width_cell= 3
app = Dash(external_stylesheets=[dbc.themes.CERULEAN]) # ref[1]

app.layout = html.Div([
    # dcc.Graph(
    dbc.Card(
        dbc.CardBody([
            dbc.Row([
                dbc.Col([html.H3('Input (Normalized)'), dcc.Graph(figure=fig_bara)] , width=width_cell), 
                # dbc.Col([html.H3('Input (Normalized)'),] , width=width_cell), 
                dbc.Col([html.H3(r'Target (Absolute)'),], width=width_cell),
                dbc.Col([html.H3(r'Prediction (Absolute)'),], width=width_cell),
                dbc.Col([html.H3(r'Error (Absolute)'),], width=width_cell),   
                ], 
            className="h-25",
            align='center'), 
            

            html.Br(),
            dbc.Row([
                # dbc.Col([vtk_view_0_html], width=width_cell),
                # dbc.Col([vtk_view_x_html, dcc.Graph(figure=fig_bara)], width=width_cell),
                dbc.Col([vtk_view_x_html], width=width_cell),
                dbc.Col([vtk_view_y_html], width=width_cell),
                dbc.Col([vtk_view_yhat_html], width=width_cell),
                dbc.Col([vtk_view_err_html], width=width_cell),
                ], 
                className="h-25",
                align='center'), 
            
            html.Br(),
            
            dbc.Row([
            #     # dbc.Col([vtk_view_0_html], width=width_cell),
                dbc.Col(create_bar([0,100] ), width=width_cell),
                dbc.Col(create_bar([0,100] ), width=width_cell*2),
            #     # dbc.Col([dcc.Graph(figure=fig_bara)], width=width_cell),
                dbc.Col(create_bar([-0.005,.02]), width=width_cell),
            ])
        ]),
        # style={"height": "100vh"},
    )
    
])

# fig = app.layout
# fig.write_image("images/fig1.png")
if __name__ == "__main__":    app.run_server(debug=True, port=8050)
# filename = 'asdf.png'
# writer = vtk.vtkOBJWriter()
# writer = vtk.vtkPNGWriter()
# writer.SetFileName(filename)
# writer.SetInputData(mesh_state_X)
# writer.Write()
