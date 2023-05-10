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
# %% References 
# https://community.plotly.com/t/how-do-you-set-page-title/40115
# help(dash.html.Title)
# https://community.plotly.com/t/colorscale-text-size/30615
# https://matplotlib.org/stable/tutorials/colors/colormaps.html
# https://github.com/plotly/dash-vtk/blob/master/docs/REFERENCES.md
# https://github.com/Kitware/vtk-js/blob/master/Sources/Rendering/Core/ColorTransferFunction/ColorMaps.json
# %% GroundTruth Class
class GroundTruth(object):
    def __init__(self, ):   pass
    def beam2D(self, vtu_file   = 'beam2d1.vtu'): 
        
        self.vtu_file   = vtu_file    
        
        self.nodes_pd   = pd.read_csv('../datasets/b2/Beam2D.vtk_nds')
        self.elements   = pd.read_csv('../datasets/b2/Beam2D.vtk_els')
        
        self.nodes      = self.nodes_pd[['X', 'Y', 'Z']]
        self.xyz        = list(self.nodes.values)

        
        self.elements           = self.elements.loc[:, self.elements.columns != 'ID']
        self.elements           = self.elements -min(self.elements.min().values)
        self.elements['count']  =  self.elements.shape[1]
        df_offset               = self.elements['count']
        del self.elements['count']
        self.df_offset          = df_offset.cumsum()     
        
        self.elements['type']   = 9 # 
        self.df_type            = self.elements['type']
        del self.elements['type']
        
        self.points_01 = [item for sublist in self.xyz for item in sublist]
    def plane(self, vtu_file   = 'plane1.vtu'): 
        
        self.vtu_file   = vtu_file    
        
        self.nodes_pd   = pd.read_csv('../datasets/pl/plane.vtk_nds')
        self.elements   = pd.read_csv('../datasets/pl/plane.vtk_els')
        
        self.nodes      = self.nodes_pd[['X', 'Y', 'Z']]
        self.xyz        = list(self.nodes.values)

        
        self.elements           = self.elements.loc[:, self.elements.columns != 'ID']
        self.elements           = self.elements -min(self.elements.min().values)
        self.elements['count']  =  self.elements.shape[1]
        df_offset               = self.elements['count']
        del self.elements['count']
        self.df_offset          = df_offset.cumsum()     
        
        self.elements['type']   = 10 # linear tetra
        self.df_type            = self.elements['type']
        del self.elements['type']
        
        self.points_01 = [item for sublist in self.xyz for item in sublist]
    def fibonacci(self, vtu_file   = 'fibonacci3D1.vtu'):       
        self.vtu_file   = vtu_file           
        self.nodes_pd   = pd.read_csv('../datasets/fs/fibonacci.vtk_nds')
        self.elements   = pd.read_csv('../datasets/fs/fibonacci.vtk_els')       
        self.nodes      = self.nodes_pd[['X', 'Y', 'Z']]
        self.xyz        = list(self.nodes.values)     
        self.elements           = self.elements.loc[:, self.elements.columns != 'ID']
        self.elements           = self.elements -min(self.elements.min().values)
        self.elements['count']  =  self.elements.shape[1]
        df_offset               = self.elements['count']
        del self.elements['count']
        self.df_offset          = df_offset.cumsum()             
        self.elements['type']   = 12 # linear hexa
        self.df_type            = self.elements['type']
        del self.elements['type']
        self.points_01 = [item for sublist in self.xyz for item in sublist]              
    def beam3D(self, vtu_file   = 'beam3D1.vtu'):         
        self.vtu_file   = vtu_file            
        # self.nodes_pd   = pd.read_csv('../evaluation/nodes_Beam3D.csv')
        self.nodes_pd   = pd.read_csv('../datasets/b3/Beam3D.vtk_nds')
        self.nodes      = self.nodes_pd[['X', 'Y', 'Z']]
        self.xyz        = list(self.nodes.values)
        self.elements           = pd.read_csv('../evaluation/elements_Beam3D.csv')
        self.elements           = self.elements.loc[:, self.elements.columns != 'ID']
        self.elements           = self.elements -min(self.elements.min().values)
        self.elements['count']  =  self.elements.shape[1]
        df_offset               = self.elements['count']
        del self.elements['count']
        self.df_offset          = df_offset.cumsum()            
        self.elements['type']   =12 # linear hexa
        self.df_type            = self.elements['type']
        del self.elements['type']        
        self.points_01 = [item for sublist in self.xyz for item in sublist]        
    def attach_result_fields(self, X, y, y_hat, err):
        self.point_values_X     =  X
        self.point_values_y     =  y
        self.point_values_y_hat =  y_hat
        self.point_values_err   =  err
    def write_results_to_vtu(self,):
        l1 = '<VTKFile type="UnstructuredGrid" version="0.1" byte_order="LittleEndian"> \n'
        l2 = '<UnstructuredGrid> \n'
        l3 = '<Piece NumberOfPoints="' + str(len(self.nodes)) + '" NumberOfCells="'+ str(len(self.elements)) +'"> \n'
        l4 = '<Points> \n'
        l5 = '<DataArray type="Float64" NumberOfComponents="3" format="ascii"> \n'
        l6 =  self.nodes.to_string(header=False, index=False) 
        l7 = '\n</DataArray> \n'
        l8 = '</Points> \n'
        l81 = '<PointData Tensors="" Vectors="" Scalars="">'
        l82 = '\n<DataArray type="Float32" Name="X_" format="ascii"> \n'
        # l83 = pd.DataFrame(point_values).to_string(header=False, index=False)
        l83 = pd.DataFrame(self.point_values_X).to_string(header=False, index=False)
        l84 = '\n</DataArray> \n'
        
        l82b = '\n<DataArray type="Float32" Name="Y_" format="ascii"> \n'
        l83b = pd.DataFrame(self.point_values_y).to_string(header=False, index=False)
        l84b = '\n</DataArray> \n'
        
        l82c = '\n<DataArray type="Float32" Name="Yhat_" format="ascii"> \n'
        l83c = pd.DataFrame(self.point_values_y_hat).to_string(header=False, index=False)
        l84c = '\n</DataArray> \n'
        
        l82d = '\n <DataArray type="Float32" Name="err_" format="ascii"> \n'
        l83d = pd.DataFrame(self.point_values_err).to_string(header=False, index=False)
        l84d = '\n</DataArray> \n'
        
        l85 = '</PointData> \n' 
        l9 = '<Cells> \n'
        l10 = '<DataArray type="Int32" Name="connectivity" format="ascii"> \n'
        l11 =  self.elements.to_string(header=False, index=False) 
        l11a = '\n</DataArray>\n'
        l12 = '\n<DataArray type="Int32" Name="offsets" format="ascii">\n'
        l13 = self.df_offset.to_string(header=False, index=False)
        l14 = '\n</DataArray>\n'
        l15 = '<DataArray type="UInt8" Name="types" format="ascii">\n'
        l16 = self.df_type.to_string(header=False, index=False) 
        l17 ='\n</DataArray>\n'
        l18 ='</Cells>\n'
        l19 ='</Piece>\n'
        l20 ='</UnstructuredGrid>\n'
        l21 ='</VTKFile>\n'
        
        self.lines = [l1, l2,l3, l4, l5, l6, l7, l8,
                 l81,
                 l82,l83,l84,         
                 l82b,l83b,l84b,
                 l82c,l83c,l84c,
                 l82d,l83d,l84d,
                 l85,           
                 l9, l10, l11,l11a,
                 l12, l13, l14, l15, l16,l17,l18,l19,l20,l21]
                # pass
        with open(self.vtu_file, 'w') as f:     f.writelines(self.lines)

            
# class Wizal(object):
    # def Wizal(self, GroudTruthClass):   
        
    def Wizal(self, ):   
        # GT = GroudTruthClass
        # GT.vtk_file_path = 'beam3D3.vtu'
        vtk_file_path = self.vtu_file
        
        reader = vtk.vtkXMLUnstructuredGridReader()
        reader.SetFileName(vtk_file_path)
        reader.Update() 
        
        self.mesh_state      = to_mesh_state(reader.GetOutput(), )
        self.mesh_state_X    = to_mesh_state(reader.GetOutput(), field_to_keep= 'X_')
        self.mesh_state_y    = to_mesh_state(reader.GetOutput(), field_to_keep= 'Y_')
        self.mesh_state_yhat = to_mesh_state(reader.GetOutput(), field_to_keep= 'Yhat_')
        self.mesh_state_err  = to_mesh_state(reader.GetOutput(), field_to_keep= 'err_')
    def generateVTKviewHTML(self, MESH_STATE, RANGE_SET = None):
        # cs_name = 'Greens'
        # cs_name =  'Blues'
        # cs_name = "Warm to Cool"
        # cs_name ="Inferno (matplotlib)"
        # cs_name = 'Brewer Diverging Spectral (9)'
        
        # cs_name = 'rdylbu'
        
        cs_name = "Cool to Warm" #  
        # cs_name = "Viridis (matplotlib)" # nefunguje
        # cs_name = 'Spectral' # nefunguje
        # self.cs_name = "PiYG"
        # self.cs_name = "RdYlBu "
        # self.cs_name = 'Brewer Diverging Spectral (9)'
        # cs_name = 'Plasma (matplotlib)'
        
        if not RANGE_SET:   rng     = [0, 10]
        else:               rng     = RANGE_SET
        vtk_view = dash_vtk.View(
            
                    [dash_vtk.GeometryRepresentation(
                        [ dash_vtk.Mesh(state = MESH_STATE), ],
                            property={
                                        "edgeVisibility": True, 
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
        # cameraPosition=[0,25,35], 
        cameraPosition=[10,10,10], 
        )
        
        return html.Div(
            style={"height": "calc(100vh - 16px)"},
            children=[html.Div(vtk_view, style={"height": "80%", "width": "100%", })]
            )

    
    def create_bar(self, range_def = [-10,10] ):
        # y = [-10,10]
        y = range_def
        min_x = -0.015
        max_x = 0.015
        min_y = -0.3
        max_y = 0.3
        
        dummy_trace=go.Scatter(x=[min_x, max_x],
                     y=[min_y, max_y],
                       # text="country",  
                     mode='markers',
                     marker=dict(
                         size=(max(y)-min(y))/100, 
                         color=[min(y), max(y)], 
                         
                         
                         colorscale= 'rdylbu_r', # reversed
                         # colorscale= 'rdylbu',
                         # colorscale= 'coolwarm',
                         # colorscale= 'Spectral',
                          # colorscale= "RdYlBu",
                         # colorscale= "PiYG",
                          # colorscale='Inferno',
                           # colorscale='plasma',  
                           # colorscale='Blues',
                          # "Inferno (matplotlib)"
                         # colorscale='Greens',  
                         
                          # colorbar=dict(thickness=100), 
                          # colorbar=dict(thickness=10, orientation='h'), 
                          colorbar=dict(thickness=20,
                                        ticklen=3, 
                                        # tickcolor='blue', 
                                        # tickcolor='orange', # funguje
                                        tickfont=dict(size=24, 
                                                       # color='blue', 
                                                       color='darkblue', 
                                                      # color='orange',  fngj
                                                      ),
                                        orientation='h'), 
                          
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

    def ViewsSet(self,):
        self.RANGE_X     =  [self.point_values_X.min(),self.point_values_X.max()]
        self.RANGE_y      = [self.point_values_y.min(),self.point_values_y.max()]
        self.RANGE_y_hat      = [self.point_values_y_hat.min(),self.point_values_y_hat.max()]
        # self.RANGE_err   =  [self.point_values_err.min(), self.point_values_err.max() ]
        self.RANGE_err   =  [self.point_values_err.min()*0.9, self.point_values_err.max()*0.9 ]
        
        self.vtk_view_0_html      =   self.generateVTKviewHTML(self.mesh_state, RANGE_SET = None)    
        self.vtk_view_x_html      =   self.generateVTKviewHTML(self.mesh_state_X, 
                                                               # RANGE_SET = None,
                                                               RANGE_SET = self.RANGE_X
                                                               )
        self.vtk_view_y_html      =   self.generateVTKviewHTML(self.mesh_state_y, 
                                                               # RANGE_SET = None,
                                                               RANGE_SET = self.RANGE_y
                                                               )
        self.vtk_view_yhat_html   =   self.generateVTKviewHTML(self.mesh_state_yhat, 
                                                               # RANGE_SET = None,
                                                               RANGE_SET = self.RANGE_y_hat,
                                                               )
        self.vtk_view_err_html    =   self.generateVTKviewHTML(self.mesh_state_err, 
                                                               # RANGE_SET = [-0.05,.05],
                                                               RANGE_SET = self.RANGE_err ,
                                                               )
    
        
    def ViewHTML_error_only(self,):
        self.ViewsSet()
        width_cell= 1
        app = Dash(external_stylesheets=[dbc.themes.CERULEAN]) # ref[1]
    
        app.layout = html.Div([
            # dcc.Graph(
            dbc.Card(
                dbc.CardBody([
                    dbc.Row([

                        dbc.Col([html.H3(r'Error (Absolute)'),], width=width_cell),   
                        ], 
                    # className="h-25",
                    align='center'), 
                    
    
                    html.Br(),
                    
                    dbc.Row([
  
                        dbc.Col([self.vtk_view_err_html],    width=width_cell),
                        ], 
                        className="h-100",
                        align='center'), 
                    
                    html.Br(),
                    
                    dbc.Row([

                        dbc.Col(self.create_bar(self.RANGE_err), width=width_cell),
                    ]
                        
                        )
                ]),
                style={"height": "100vh"},
            )
            
        ])
        return app
        if __name__ == "__main__":    app.run_server(debug=True, port=8050)
    def ViewHTML(self,):
        self.RANGE_X     =  [self.point_values_X.min(),self.point_values_X.max()]
        self.RANGE_y      = [self.point_values_y.min(),self.point_values_y.max()]
        self.RANGE_y_hat      = [self.point_values_y_hat.min(),self.point_values_y_hat.max()]
        # self.RANGE_err   =  [self.point_values_err.min(), self.point_values_err.max() ]
        self.RANGE_err   =  [self.point_values_err.min()*0.9, self.point_values_err.max()*0.9 ]
        
        self.vtk_view_0_html      =   self.generateVTKviewHTML(self.mesh_state, RANGE_SET = None)    
        self.vtk_view_x_html      =   self.generateVTKviewHTML(self.mesh_state_X, 
                                                               # RANGE_SET = None,
                                                               RANGE_SET = self.RANGE_X
                                                               )
        self.vtk_view_y_html      =   self.generateVTKviewHTML(self.mesh_state_y, 
                                                               # RANGE_SET = None,
                                                               RANGE_SET = self.RANGE_y
                                                               )
        self.vtk_view_yhat_html   =   self.generateVTKviewHTML(self.mesh_state_yhat, 
                                                               # RANGE_SET = None,
                                                               RANGE_SET = self.RANGE_y_hat,
                                                               )
        self.vtk_view_err_html    =   self.generateVTKviewHTML(self.mesh_state_err, 
                                                               # RANGE_SET = [-0.05,.05],
                                                               RANGE_SET = self.RANGE_err ,
                                                               )
        
        width_cell= 3
        app = Dash(external_stylesheets=[dbc.themes.CERULEAN]) # ref[1]
    
        app.layout = html.Div([
            html.Title('Test'),
            # help(dash.html.Title)
            # dcc.Graph(
            dbc.Card(
                dbc.CardBody([
                    dbc.Row([
                        dbc.Col([html.H3('Input (Normalized)'), ] , width=width_cell), 
                        # dbc.Col([html.H3('Input (Normalized)'),] , width=width_cell), 
                        dbc.Col([html.H3(r'Target (Absolute)'),], width=width_cell),
                        dbc.Col([html.H3(r'Prediction (Absolute)'),], width=width_cell),
                        dbc.Col([html.H3(r'Error (Absolute)'),], width=width_cell),   
                        ], 
                    # className="h-25",
                    align='center'), 
                    
    
                    html.Br(),
                    
                    dbc.Row([
                        # dbc.Col([vtk_view_0_html], width=width_cell),
                        # dbc.Col([vtk_view_x_html, dcc.Graph(figure=fig_bara)], width=width_cell),
                        dbc.Col([self.vtk_view_x_html],      width=width_cell),
                        dbc.Col([self.vtk_view_y_html],      width=width_cell),
                        dbc.Col([self.vtk_view_yhat_html],   width=width_cell),
                        dbc.Col([self.vtk_view_err_html],    width=width_cell),
                        ], 
                        className="h-25",
                        align='center'), 
                    
                    html.Br(),
                    
                    dbc.Row([
                    #     # dbc.Col([vtk_view_0_html], width=width_cell),
                        dbc.Col(self.create_bar(self.RANGE_X ), width=width_cell),
                        
                        dbc.Col(self.create_bar(self.RANGE_y ), width=width_cell*2),
                    #     # dbc.Col([dcc.Graph(figure=fig_bara)], width=width_cell),
                        dbc.Col(self.create_bar(self.RANGE_err), width=width_cell),
                    ]
                        
                        )
                ]),
                style={"height": "100vh"},
            )
            
        ])
        return app
        # if __name__ == "__main__":    app.run_server(debug=True, port=8050)
        
        
''' colorscales notes
Invalid value of type 'builtins.str' received for the 'colorscale' property of scatter.marker
    Received value: 'coolwarm'

The 'colorscale' property is a colorscale and may be
specified as:
  - A list of colors that will be spaced evenly to create the colorscale.
    Many predefined colorscale lists are included in the sequential, diverging,
    and cyclical modules in the plotly.colors package.
  - A list of 2-element lists where the first element is the
    normalized color level value (starting at 0 and ending at 1),
    and the second item is a valid color string.
    (e.g. [[0, 'green'], [0.5, 'red'], [1.0, 'rgb(0, 0, 255)']])
  - One of the following named colorscales:
        ['aggrnyl', 'agsunset', 'algae', 'amp', 'armyrose', 'balance',
         'blackbody', 'bluered', 'blues', 'blugrn', 'bluyl', 'brbg',
         'brwnyl', 'bugn', 'bupu', 'burg', 'burgyl', 'cividis', 'curl',
         'darkmint', 'deep', 'delta', 'dense', 'earth', 'edge', 'electric',
         'emrld', 'fall', 'geyser', 'gnbu', 'gray', 'greens', 'greys',
         'haline', 'hot', 'hsv', 'ice', 'icefire', 'inferno', 'jet',
         'magenta', 'magma', 'matter', 'mint', 'mrybm', 'mygbm', 'oranges',
         'orrd', 'oryel', 'oxy', 'peach', 'phase', 'picnic', 'pinkyl',
         'piyg', 'plasma', 'plotly3', 'portland', 'prgn', 'pubu', 'pubugn',
         'puor', 'purd', 'purp', 'purples', 'purpor', 'rainbow', 'rdbu',
         'rdgy', 'rdpu', 'rdylbu', 'rdylgn', 'redor', 'reds', 'solar',
         'spectral', 'speed', 'sunset', 'sunsetdark', 'teal', 'tealgrn',
         'tealrose', 'tempo', 'temps', 'thermal', 'tropic', 'turbid',
         'turbo', 'twilight', 'viridis', 'ylgn', 'ylgnbu', 'ylorbr',
         'ylorrd'].
    Appending '_r' to a named colorscale reverses it.
'''        
