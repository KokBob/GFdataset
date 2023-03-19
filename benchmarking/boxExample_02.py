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

df = pd.read_csv("violin_gf_2.csv")

# import seaborn as sns
# sns.set_theme(style="ticks", palette="pastel")

# Load the example tips dataset
# tips = sns.load_dataset("tips")

# Draw a nested boxplot to show bills by day and time
# sns.boxplot(x="S", y="M",
#             hue="T", palette=["m", "g"],
#             data=df)
# sns.despine(offset=10, trim=True)

app = dash.Dash(__name__)
server = app.server


fig = go.Figure()
fig = px.box(df,x="S", y="M", color="T")


# fig.update_traces(meanline_visible=True,
#                   points='all', # show all points
#                   jitter=0.05,  # add some jitter on points for better visibility
#                   scalemode='count',) #scale violin plot area with total count
# fig.update_layout(
#     title_text="Total bill distribution<br><i>scaled by number of bills per gender",
#     violingap=0, violingroupgap=0, violinmode='overlay')

fig.update_yaxes(type="log")
fig.show()

app.layout = html.Div(children=[
    html.H1(children='VIOLIN '),
    dcc.Graph(id='ts original/smoothed',figure=fig),
])
if __name__ == "__main__":    app.run_server(debug=True)

