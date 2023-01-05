# -*- coding: utf-8 -*-
"""
https://plotly.com/python/box-plots/
"""
import plotly.graph_objects as go
import dash
from dash import html 
from dash import dcc
import pandas as pd
import plotly.express as px
# df = pd.read_csv("violin_gf_2.csv")
df = pd.read_csv("violin_gf_03.csv")

app = dash.Dash(__name__)
server = app.server

fig = px.box(df,x="S", y="M", color="T")

fig.update_yaxes(type="log")
fig.show()

app.layout = html.Div(children=[
    html.H1(children='VIOLIN '),
    dcc.Graph(id='ts original/smoothed',figure=fig),
])
if __name__ == "__main__":    app.run_server(debug=True)

