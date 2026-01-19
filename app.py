import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import dash_bootstrap_components as dbc
import pandas as pd
import numpy as np

# --- ENHANCED MOCK MODEL ---
class MockModel:
    def predict(self, X):
        # Base calculation
        base_price = 30000
        area_coef = 2200  # $2200 per m2
        
        # Neighborhood Multipliers (To make the app feel real)
        # Puerto Madero is expensive (2.5x), Boca is cheaper (0.9x)
        hood_factors = {
            'Puerto Madero': 2.5,
            'Palermo': 1.5,
            'Recoleta': 1.4,
            'Belgrano': 1.3,
            'San Telmo': 1.1,
            'Boca': 0.9,
            'Caballito': 1.0
        }
        
        # Get the multiplier for the selected neighborhood (default to 1.0)
        hood = X['neighborhood'].values[0]
        multiplier = hood_factors.get(hood, 1.0)
        
        # Calculate price
        # Price = (Base + (Area * Cost)) * Neighborhood_Factor
        pred = (base_price + (X['surface_covered_in_m2'] * area_coef)) * multiplier
        return np.array([pred])

model = MockModel()

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.LUX])
server = app.server

# --- LAYOUT ---
app.layout = dbc.Container([
    # Title
    dbc.Row([
        dbc.Col(html.H1("Buenos Aires Real Estate Predictor", className="text-center text-primary mb-4"), width=12)
    ], className="mt-5"),

    dbc.Row([
        # Left Column: Inputs
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Property Details", className="fw-bold"),
                dbc.CardBody([
                    
                    # 1. Neighborhood
                    html.Label("Neighborhood"),
                    dcc.Dropdown(
                        id='neighborhood-input',
                        options=[
                            {'label': 'Puerto Madero (Premium)', 'value': 'Puerto Madero'},
                            {'label': 'Palermo', 'value': 'Palermo'},
                            {'label': 'Recoleta', 'value': 'Recoleta'},
                            {'label': 'Belgrano', 'value': 'Belgrano'},
                            {'label': 'San Telmo', 'value': 'San Telmo'},
                            {'label': 'Caballito', 'value': 'Caballito'},
                            {'label': 'Boca', 'value': 'Boca'},
                        ],
                        value='Palermo',
                        className="mb-3"
                    ),

                    # 2. Area
                    html.Label("Surface Area (m²)"),
                    dcc.Slider(
                        id='area-input',
                        min=20, max=200, step=5, value=50,
                        marks={i: f'{i}m²' for i in range(20, 201, 50)},
                        className="mb-3"
                    ),

                    # 3. Lat/Lon (Now Visible in a Grid)
                    dbc.Row([
                        dbc.Col([
                            html.Label("Latitude"),
                            dcc.Input(id='lat-input', type='number', value=-34.60, step=0.001, className="form-control")
                        ], width=6),
                        dbc.Col([
                            html.Label("Longitude"),
                            dcc.Input(id='lon-input', type='number', value=-58.38, step=0.001, className="form-control")
                        ], width=6),
                    ], className="mb-3"),
                ])
            ], className="shadow-sm")
        ], width=12, md=4),

        # Right Column: Prediction
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H4("Estimated Value (USD)", className="text-center text-muted"),
                    html.Div(id='prediction-output', className="display-4 text-center text-success fw-bold my-5"),
                    html.P("Price adjusts based on neighborhood premium and surface area.", className="text-center small text-muted")
                ])
            ], className="shadow-sm h-100")
        ], width=12, md=8)
    ])
], fluid=True)

# --- CALLBACK ---
@app.callback(
    Output('prediction-output', 'children'),
    [Input('area-input', 'value'),
     Input('lat-input', 'value'),
     Input('lon-input', 'value'),
     Input('neighborhood-input', 'value')]
)
def update_prediction(area, lat, lon, neighborhood):
    # Create DataFrame for the mock model
    input_data = pd.DataFrame({
        'surface_covered_in_m2': [area],
        'lat': [lat],
        'lon': [lon],
        'neighborhood': [neighborhood]
    })
    
    try:
        prediction = model.predict(input_data)[0]
        return f"${prediction:,.2f}"
    except Exception as e:
        return "Calculating..."

if __name__ == '__main__':
    app.run_server(debug=True)
