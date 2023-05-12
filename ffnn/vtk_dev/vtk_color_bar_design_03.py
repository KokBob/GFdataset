import plotly.graph_objects as go # or plotly.express as px
fig = go.Figure() # or any Plotly Express function e.g. px.bar(...)
# fig.add_trace( ... )
# fig.update_layout( ... )
# https://stackoverflow.com/questions/55447131/how-to-add-a-colorbar-to-an-already-existing-plotly-figure
import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly.express as px


colorbar_trace  = go.Scatter(x=[None],
                             y=[None],
                             mode='markers',
                             marker=dict(
                                  # colorscale='blues', 
                                   colorscale='plasma', 
                                 showscale=True,
                                 cmin=-4.5,
                                 cmax=4.5,
                                 colorbar=dict(thickness=1050, 
                                               # tickvals=[-5, 0, 5], ticktext=['Low', 'Medium', 'High'],
                                                tickvals=[-5, 5], ticktext=['Low', 'High'], 
                                               outlinewidth=0)
                             ),
                             hoverinfo='none'
                            )

fig['layout']['showlegend'] = False
fig['layout']['showlegend'] = True
fig.add_trace(colorbar_trace)

# y = data['y']
# y = [-1,1]
# data['marker'] = dict(color=y, colorscale=red_blue, colorbar=dict(thickness=10))
# fig = go.Figure(data=[data], layout=layout)
# fig = px.colors.diverging.swatches_continuous()

# fig.show()

app = dash.Dash()
app.layout = html.Div([
    dcc.Graph(figure=fig)
])

if __name__ == "__main__":    app.run_server(debug=True, port=8050)