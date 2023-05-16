# -*- coding: utf-8 -*-
"""
https://stackoverflow.com/questions/60231146/how-can-i-turn-my-dataframe-into-a-radar-chart-using-python
"""
import plotly.graph_objects as go
from dash import Dash, html, dcc
import plotly.express as px
import pandas as pd

categories = ['accuracucy','mechanical properties','chemical stability',
              'thermal stability', 'device integration']

def example1wrapped():
    fig = go.Figure()
    # df = pd.DataFrame(dict(
    #     r=[1, 5, 2, 2, 3],
    #     theta=['processing cost','mechanical properties','chemical stability',
    #            'thermal stability', 'device integration']))
    
    # fig = px.line_polar(df, r='r', theta='theta', line_close=True)
    fig.add_trace(go.Scatterpolar(
          r=[.95, 5, 2, 2, 3],
          theta=categories,
           fill='toself',
          name='Product A'
    ))
    fig.add_trace(go.Scatterpolar(
          r=[.93, 3, 2.5, 1, 2],
          theta=categories,
           fill='toself',
          name='Product B'
    ))
    
    fig.add_trace(go.Scatterpolar(
          r=[.93, 3, 1, 1, 2],
          theta=categories,
           fill='toself',
          name='Product C',
           # line_close=True
    ))
    
    fig.update_layout(
      polar=dict(
        radialaxis=dict(
          visible=True,
          range=[0, 5]
        )),
      showlegend=False
    )
    return fig
def example2wrapped():
    theta_variables = ['processing cost', 'mechanical properties', 'chemical stability',
           'thermal stability', 'device integration', 'next']
    # ref https://stackoverflow.com/questions/73269460/add-multiple-lines-in-radar-plot-python-plotly

    df1 = pd.DataFrame(dict(
        r=[1, 5, 2, 2, 3, 2],
        theta=theta_variables))
    
    
    df2 = pd.DataFrame(dict(
        r=[.1, .5, .2, .2, .3, 1],
        theta=theta_variables))
    
    df3 = pd.DataFrame(dict(
        r=[1.7, 4.5, .82, .9, .05, 1],
        theta=theta_variables))
    
    df4 = pd.DataFrame(dict(
        r=[17, 4.5, .82, .9, 2.05, 3],
        theta=theta_variables))
    
    
    df1['Model'] = 'LR'
    df2['Model'] = 'FN'
    df3['Model'] = 'GCN'
    df4['Model'] = 'SAGE'
    df = pd.concat([df1, df2, df3, df4], axis=0)
    
    fig = px.line_polar(df, r='r', color='Model', theta='theta', line_close=True)
    return fig

# fig2plot = example1wrapped()
fig2plot = example2wrapped()
app = Dash(__name__)
app.layout = html.Div(children=[
    html.H1(children='Hello Dash'),

    html.Div(children='''
        Dash: A web application framework for your data.
    '''),

    dcc.Graph(
        id='example-graph',
        figure=fig2plot
    )
])

if __name__ == '__main__':    app.run_server(debug=True)
