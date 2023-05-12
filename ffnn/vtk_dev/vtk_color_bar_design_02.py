import plotly.graph_objects as go # or plotly.express as px
fig = go.Figure() # or any Plotly Express function e.g. px.bar(...)
# fig.add_trace( ... )
# fig.update_layout( ... )

import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly.express as px


fig = px.colors.diverging.swatches_continuous()

# fig.show()

app = dash.Dash()
app.layout = html.Div([
    dcc.Graph(figure=fig)
])

if __name__ == "__main__":    app.run_server(debug=True, port=8050)