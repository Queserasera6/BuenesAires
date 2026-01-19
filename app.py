import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import dash_bootstrap_components as dbc
import pandas as pd
import numpy as np

# --- ENHANCED MOCK MODEL (FIXED) ---
class MockModel:
    def predict(self, X):
        # FIX: Use .iloc[0] to ensure we get a single SCALAR number, not a Series
        area = float(X['surface_covered_in_m2'].iloc[0])
        hood = X['neighborhood'].iloc[0]
        
        # Base calculation
        base_price = 30000
        area_coef = 2200
        
        # Neighborhood Multipliers
        hood_factors = {
            'Puerto Madero': 2.5,
            'Palermo': 1.5,
            'Recoleta': 1.4,
            'Belgrano': 1.3,
            'San Telmo': 1.1,
            'Boca': 0.9,
            'Caballito': 1.0
        }
        
        multiplier = hood_factors.get(hood, 1.0)
        
        # Calculate price (Now strictly scalar math)
        pred = (base_price + (area * area_coef)) * multiplier
        
        # Return as a list so [0] access works
        return [pred]

model = MockModel()

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.LUX])
server = app.server

# --- LAYOUT ---
app.layout = dbc.Container([
    dbc.Row([
        dbc.Col(html.H1("Buenos Aires Real Estate Predictor", className="text-center text-primary mb-4"), width=12)
    ], className="mt-5"),

    dbc.Row([
        # Left Column: Inputs
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Property Details", className="fw-bold"),
                dbc.CardBody([
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
                    html.Label("Surface Area (m²)"),
                    dcc.Slider(
                        id='area-input',
                        min=20, max=200, step=5, value=50,
                        marks={i: f'{i}m²' for i in range(20, 201, 50)},
                        className="mb-3"
                    ),
                    # Hidden Lat/Lon for simplicity in demo
                    dcc.Input(id='lat-input', type='hidden', value=-34.60),
                    dcc.Input(id='lon-input', type='hidden', value=-58.38),
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
    # Create DataFrame
    input_data = pd.DataFrame({
        'surface_covered_in_m2': [area],
        'lat': [lat],
        'lon': [lon],
        'neighborhood': [neighborhood]
    })
    
    try:
        # Predict
        prediction = model.predict(input_data)[0]
        return f"${prediction:,.2f}"
    except Exception as e:
        # If error, print it to the screen so we can see it!
        return f"Error: {str(e)}"

if __name__ == '__main__':
    app.run_server(debug=True)
