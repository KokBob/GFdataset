import plotly.graph_objects as go # or plotly.express as px
fig = go.Figure() # or any Plotly Express function e.g. px.bar(...)
# fig.add_trace( ... )
# fig.update_layout( ... )
# https://stackoverflow.com/questions/55447131/how-to-add-a-colorbar-to-an-already-existing-plotly-figure
# https://plotly.com/python/colorscales/
# https://stackoverflow.com/questions/29968152/setting-background-color-to-transparent-in-plotly-plots
import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly.express as px


def create_bar():
    y = [-10,10]
    min_x = -0.015
    max_x = 0.015
    min_y = -0.3
    max_y = 0.3
    
    dummy_trace=go.Scatter(x=[min_x, max_x],
                 y=[min_y, max_y],
                 mode='markers',
                 marker=dict(
                     size=(max(y)-min(y))/100, 
                     color=[min(y), max(y)], 
                     colorscale='plasma',  
                     # colorbar=dict(thickness=100), 
                     colorbar=dict(thickness=100, orientation='h'), 
                     showscale=True
                 ),
                 hoverinfo='none',
                 # plot_bgcolor='rgba(0,0,0,0)'
                )
    
    layout = dict(xaxis=dict(visible=False), yaxis=dict(visible=False), 
                  # paper_bgcolor='rgb(233,233,233)',
                  plot_bgcolor='rgba(0,0,0,0)'
                  )
    fig_bar = go.Figure([dummy_trace], layout)
    return fig_bar

fig = create_bar()
app = dash.Dash()
app.layout = html.Div([
    dcc.Graph(figure=fig)
])

if __name__ == "__main__":    app.run_server(debug=True, port=8050)