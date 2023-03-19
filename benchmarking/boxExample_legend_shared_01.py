# -*- coding: utf-8 -*-
"""
https://plotly.com/python/box-plots/
https://stackoverflow.com/questions/47932781/plotly-python-subplots-padding-facets-and-sharing-legends
https://stackoverflow.com/questions/68503614/multiple-boxplots-in-subplots-with-plotly
"""
import plotly.graph_objects as go
import dash
from dash import html 
from dash import dcc
import pandas as pd
import plotly.express as px
from plotly.subplots import make_subplots

# df = pd.read_csv("violin_gf_13.csv")

app = dash.Dash(__name__)
# fig = make_subplots(
#     rows=1, cols=2, subplot_titles=[c for c in df.columns if "var" in c]
# )
# fig1 = px.box(df,x="IDS", y="MSE", color="METHOD")
# fig2 = px.box(df,x="IDS", y="T_end", color="METHOD")

# fig1.update_yaxes(type="log", title = r"$MSE$ [MPa/sample]")
# fig2.update_yaxes(type="log", title = r"$t_{end} [s]$")


import pandas as pd
import numpy as np
import plotly.express as px
from plotly.subplots import make_subplots

df = pd.DataFrame({**{"month":np.tile(pd.date_range("1-jul-2020", freq="M", periods=12),20)[0:200],
                   "tipo":np.random.choice(["N","S"],200)},
                   **{f"var{v}":np.random.uniform(2,5,200) for v in range(8)}})

fig = make_subplots(
    rows=3, cols=3, subplot_titles=[c for c in df.columns if "var" in c]
)
app.layout = html.Div(children=[
    html.H1(children='VIOLIN '),
    html.Div(children=''' MSE of validation batch  '''),

    
    dcc.Graph(id='mse', figure=fig, style={'display': 'inline-block'}),

])
for v in range(8):
    for t in px.box(df, x="month", y=f"var{v}", color="tipo").data:
        fig.add_trace(t, row=(v//3)+1, col=(v%3)+1)

# modifications needed to fix up display of sub-plots
fig.update_layout(
    boxmode="group", margin={"l": 0, "r": 0, "t": 20, "b": 0}
).update_traces(showlegend=False, selector=lambda t: "var0" not in t.hovertemplate)



if __name__ == "__main__":    app.run_server(debug=True)