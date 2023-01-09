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
# df = pd.read_csv("violin_gf_03.csv")
# df = pd.read_csv("violin_gf_04.csv")
# df = pd.read_csv("violin_gf_07.csv")
# df = pd.read_csv("violin_gf_08.csv")
# df = pd.read_csv("violin_gf_08.csv")
# df = pd.read_csv("violin_gf_09.csv")
# df = pd.read_csv("violin_gf_10.csv")
# df = pd.read_csv("violin_gf_11.csv")

df = pd.read_csv("violin_gf_12.csv")

app = dash.Dash(__name__)
server = app.server



# fig = px.box(df,x="S", y="M", color="T")
fig = px.box(df,x="IDS", y="MSE", color="METHOD")
# fig = px.box(df,x="IDS", y="T_end", color="METHOD")

fig.update_xaxes( title = r"$ IDS $")
fig.update_yaxes(type="log", title = r"$MSE$ [MPa/sample]")
# fig.update_yaxes(type="log", title = r"$t_{end} [s]$")
# fig.update_xaxes(labels = ['b2','b2','b2','b2'])
# fig.update_layout(plot_bgcolor='rgb(10,10,10)')
fig.update_layout(plot_bgcolor='rgb(255,255,255)')
# fig.update_xaxes(color='black')
# fig.update_yaxes(color='black')
fig.show()

app.layout = html.Div(children=[
    html.H1(children='VIOLIN '),
    dcc.Graph(id='ts original/smoothed',figure=fig),
])

# fig.update_layout(
#     # title="Plot Title",
#     # xaxis_title="X Axis Title",
#     yaxis_title=r"t_{end} [s]",
#     legend_title="Methods",
#     # font=dict(
#     #     family="Courier New, monospace",
#     #     size=18,
#     #     color="RebeccaPurple"
#     # )
# )

# if __name__ == "__main__":    app.run_server(debug=True)

# %%

# df_gb_METHOD = df.groupby('METHOD')
df_gb_IDSMETHOD = df.groupby(['IDS','METHOD'])
# %%
# https://www.itl.nist.gov/div898/handbook/eda/section3/dexsdplo.htm
# https://www.statology.org/pandas-groupby-describe/
# https://towardsdatascience.com/accessing-data-in-a-multiindex-dataframe-in-pandas-569e8767201d
# https://medium.com/codex/say-goodbye-to-loops-in-python-and-welcome-vectorization-e4df66615a52
# dfDES = pd.DataFrame(df_gb_IDSMETHOD.describe())
# %%
dfDES = pd.DataFrame(df_gb_IDSMETHOD.describe()[['MSE','T_end']])
# %%
table_MSE = pd.DataFrame([dfDES['MSE','mean'], dfDES['MSE','std']])
# table_MSE.to_csv('MSE_summary_gf_11.csv')
# table_Tend = pd.DataFrame([dfDES['T_end','mean'], dfDES['T_end','std']])
# table_Tend.to_csv('Tend_summary_gf_11a.csv',float_format="%.2e")
# %%
# table_Tend.groupby('IDS').agg('mean')
dsumm = pd.DataFrame(columns=['b2', 'b3', 'fs0', 'pl'], index=['LR', 'FN', 'GCN0', 'SAGE0'])
eval_ = table_MSE # table_Tend
for ids_, mtd_ in eval_:
    d_ = eval_[ids_]
    for _ in d_:
      
        s1 = "%.0E" %d_[_][0]
        s2 = "%.0E" %d_[_][1]
        sa = s1 + '±' + s2    
        try: dsumm.loc[_][ids_] = sa
        except: pass

dsumm.to_csv('mse_summary_table.csv')
    
    
