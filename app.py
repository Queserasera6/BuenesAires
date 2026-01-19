import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import dash_bootstrap_components as dbc
import pandas as pd
import numpy as np

# Mock Model for production demo
class MockModel:
    def predict(self, X):
        base_price = 50000
        area_coef = 2500
        pred = base_price + (X['surface_covered_in_m2'] * area_coef)
        return pred.values

model = MockModel()

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.LUX])
server = app.server  # <--- THIS IS CRITICAL FOR DEPLOYMENT

app.layout = dbc.Container([
    dbc.Row([dbc.Col(html.H1("Buenos Aires Real Estate Predictor", className="text-center text-primary mb-4"), width=12)], className="mt-5"),
    dbc.Row([
        dbc.Col([
            dbc.Card([dbc.CardBody([
                html.Label("Surface Area (m²)"),
                dcc.Slider(id='area-input', min=20, max=200, step=5, value=50, marks={i: f'{i}m²' for i in range(20, 201, 40)}, className="mb-3"),
                html.Label("Neighborhood"),
                dcc.Dropdown(id='neighborhood-input', options=[{'label': 'Palermo', 'value': 'Palermo'}, {'label': 'Recoleta', 'value': 'Recoleta'}], value='Palermo', className="mb-3"),
                # Hidden inputs for layout consistency
                dcc.Input(id='lat-input', type='hidden', value=-34.60),
                dcc.Input(id='lon-input', type='hidden', value=-58.38),
            ])])
        ], width=12, md=4),
        dbc.Col([
            dbc.Card([dbc.CardBody([
                html.H4("Estimated Value", className="text-center"),
                html.Div(id='prediction-output', className="display-4 text-center text-success fw-bold my-4")
            ])], className="h-100")
        ], width=12, md=8)
    ])
], fluid=True)

@app.callback(
    Output('prediction-output', 'children'),
    [Input('area-input', 'value'), Input('lat-input', 'value'), Input('lon-input', 'value'), Input('neighborhood-input', 'value')]
)
def update_prediction(area, lat, lon, neighborhood):
    input_data = pd.DataFrame({'surface_covered_in_m2': [area], 'lat': [lat], 'lon': [lon], 'neighborhood': [neighborhood]})
    return f"${model.predict(input_data)[0]:,.2f}"

if __name__ == '__main__':
    app.run_server(debug=True)
