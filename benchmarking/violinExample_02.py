# -*- coding: utf-8 -*-
"""
"""
import plotly.graph_objects as go
import dash
from dash import html 
from dash import dcc
import pandas as pd

df = pd.read_csv("violin_gf_01.csv")

pointpos_male = [-0.9,-1.1]
pointpos_female = [0.45,0.55]
# pointpos_male = [-0.9,-1.1,-0.6,-0.3]
# pointpos_female = [0.45,0.55,1,0.4]
# show_legend = [True,False,False,False]
show_legend = [True,False]

app = dash.Dash(__name__)
server = app.server

fig = go.Figure()

for i in range(0,len(pd.unique(df['S']))):
    fig.add_trace(go.Violin(x=df['S'][(df['T'] == 'GCN0') &
                                        (df['S'] == pd.unique(df['S'])[i])],
                            y=df['M'][(df['T'] == 'GCN0')&
                                               (df['S'] == pd.unique(df['S'])[i])],
                            legendgroup='M', scalegroup='M', name='M',
                            side='negative',
                            pointpos=pointpos_male[i], # where to position points
                            line_color='lightseagreen',
                            showlegend=show_legend[i],
                            # box=True,
                            # log_y=True
                            )
             )
    fig.add_trace(go.Violin(x=df['S'][(df['T'] == 'SAGE0') &
                                        (df['S'] == pd.unique(df['S'])[i])],
                            y=df['M'][(df['T'] == 'SAGE0')&
                                               (df['S'] == pd.unique(df['S'])[i])],
                            legendgroup='F', scalegroup='F', name='F',
                            side='positive',
                            
                            pointpos=pointpos_female[i],
                            line_color='mediumpurple',
                            showlegend=show_legend[i],
                            
                            # box=True,
                            # log_y=True
                            )
             )
box=True
# update characteristics shared by all traces
fig.update_traces(meanline_visible=True,
                  points='all', # show all points
                  jitter=0.05,  # add some jitter on points for better visibility
                  scalemode='count',) #scale violin plot area with total count
fig.update_layout(
    title_text="Total bill distribution<br><i>scaled by number of bills per gender",
    violingap=0, violingroupgap=0, violinmode='overlay')

fig.update_yaxes(type="log")

app.layout = html.Div(children=[
    html.H1(children='VIOLIN '),
    dcc.Graph(id='ts original/smoothed',figure=fig),
])
if __name__ == "__main__":    app.run_server(debug=True)