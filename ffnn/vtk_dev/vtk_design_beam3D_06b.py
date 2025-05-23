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
import plotly.io as pio
import plotly.graph_objects as go
# plotly.offline.plot(vtk_view_y, filename='./vtk_y.html')
# https://kitware.github.io/vtk-examples/site/VTKBook/08Chapter8/
# https://discourse.vtk.org/t/vtkimagedata-to-png/8418/3
# https://dash-bootstrap-components.opensource.faculty.ai/docs/themes/
# ref[0]> https://towardsdatascience.com/3d-mesh-models-in-the-browser-using-python-dash-vtk-e15cbf36a132
# ref[0]> https://bootswatch.com/
# https://www.mecharithm.com/latex-to-html/
# vtk_file_path = 'beam36.vtu'
# vtk_file_path = 'beam3D1.vtu'

def make_colorbar(title, rng, bgnd='rgb(51, 76, 102)'):
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(x=[None],
                   y=[None],
                   mode='markers',
                   marker=dict(
                       colorscale='plasma',
                       showscale=True,
                       cmin=rng[0],
                       cmax=rng[1],
                       colorbar=dict(
                           title_text=title, 
                           title_font_color='white', 
                           title_side='top',
                           thicknessmode="pixels", thickness=50,
                           #  lenmode="pixels", len=200,
                           yanchor="middle", y=0.5, ypad=10,
                           xanchor="left", x=0., xpad=10,
                           ticks="outside",
                           tickcolor='white',
                           tickfont={'color':'white'}
                           #  dtick=5
                       )
        ),
            hoverinfo='none'
        )
    )
    fig.update_layout(width=150, margin={'b': 0, 'l': 0, 'r': 0, 't': 0}, autosize=False, plot_bgcolor=bgnd)
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    return fig

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

fig_ = make_colorbar('title', rng, bgnd='rgb(51, 76, 102)')
def generateVTKviewHTML(MESH_STATE, fig__, RANGE_SET = None, ):
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
                        # showScalarBar=  True # funguje yto NE
                        )
                    , fig__],
    pickingModes=["hover"],
    background=[1, 1, 1],
    cameraPosition=[10,25,35], 
    )
    
    return html.Div(
        style={"height": "calc(100vh - 16px)"},
        children=[html.Div(vtk_view, style={"height": "80%", "width": "100%", })]
        )
    
vtk_view_0_html      =   generateVTKviewHTML(mesh_state,fig_ ,RANGE_SET = None)    
vtk_view_x_html      =   generateVTKviewHTML(mesh_state_X,fig_ , RANGE_SET = [0,1])
vtk_view_y_html      =   generateVTKviewHTML(mesh_state_y,fig_ , RANGE_SET = None)
vtk_view_yhat_html   =   generateVTKviewHTML(mesh_state_yhat,fig_ , RANGE_SET = None)
vtk_view_err_html    =   generateVTKviewHTML(mesh_state_err,fig_ , RANGE_SET = [-0.005,.02])


width_cell= 3
app = Dash(external_stylesheets=[dbc.themes.CERULEAN]) # ref[1]

app.layout = html.Div([
    # dcc.Graph(
    dbc.Card(
        dbc.CardBody([
            dbc.Row([
                dbc.Col([html.H3('Input (Normalized)'),], width=width_cell), 
                dbc.Col([html.H3(r'Target (Absolute)'),], width=width_cell),
                dbc.Col([html.H3(r'Prediction (Absolute)'),], width=width_cell),
                dbc.Col([html.H3(r'Error (Absolute)'),], width=width_cell),   
                ], 
            align='center'), 

            html.Br(),
            dbc.Row([
                # dbc.Col([vtk_view_0_html], width=width_cell),
                dbc.Col([vtk_view_x_html], width=width_cell),
                dbc.Col([vtk_view_y_html], width=width_cell),
                dbc.Col([vtk_view_yhat_html], width=width_cell),
                dbc.Col([vtk_view_err_html], width=width_cell),
            ])
        ])
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
