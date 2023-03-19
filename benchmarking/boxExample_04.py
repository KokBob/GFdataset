# -*- coding: utf-8 -*-
"""
https://plotly.com/python/box-plots/
https://stackoverflow.com/questions/47932781/plotly-python-subplots-padding-facets-and-sharing-legends
"""
import plotly.graph_objects as go
import dash
from dash import html 
from dash import dcc
import pandas as pd
import plotly.express as px
from plotly.subplots import make_subplots

df = pd.read_csv("violin_gf_13.csv")

app = dash.Dash(__name__)

fig1 = px.box(df,x="IDS", y="MSE", color="METHOD")
fig2 = px.box(df,x="IDS", y="T_end", color="METHOD")

fig1.update_yaxes(type="log", title = r"$MSE$ [MPa/sample]")
fig2.update_yaxes(type="log", title = r"$t_{end} [s]$")


app.layout = html.Div(children=[
    html.H1(children='VIOLIN '),
    html.Div(children=''' MSE of validation batch  '''),
    html.Div(children=''' time elapsed validation batch  '''),
    dcc.Graph(id='mse', figure=fig1, style={'display': 'inline-block'}),
    dcc.Graph(id='tend',figure=fig2, style={'display': 'inline-block'}),
])





if __name__ == "__main__":    app.run_server(debug=True)