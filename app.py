import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import dash_bootstrap_components as dbc
import pandas as pd
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import Ridge
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

# --- MODEL INITIALIZATION & SIMULATION ---
def train_and_evaluate():
    """
    Initializes the Ridge Regression pipeline and trains it on a simulated 
    housing dataset. 
    """
    np.random.seed(42)
    n_samples = 1500 
    
    # 1. Initialize Feature Space (Buenos Aires Region)
    areas = np.random.randint(30, 250, n_samples)
    lats = np.random.uniform(-34.65, -34.55, n_samples)
    lons = np.random.uniform(-58.50, -58.35, n_samples)
    neighborhoods = np.random.choice(
        ['Palermo', 'Recoleta', 'Belgrano', 'San Telmo', 'Caballito', 'Boca', 'Puerto Madero'], 
        n_samples
    )
    
    df = pd.DataFrame({
        'surface_covered_in_m2': areas,
        'lat': lats,
        'lon': lons,
        'neighborhood': neighborhoods
    })
    
    # 2. Define Market Logic (Ground Truth)
    # BASE PRICE increased to ensuring no negative start
    base_price = 50000 
    
    hood_premiums = {
        'Puerto Madero': 120000, 'Palermo': 60000, 'Recoleta': 55000,
        'Belgrano': 45000, 'San Telmo': 25000, 'Caballito': 30000, 'Boca': 8000
    }
    
    prices = []
    for i in range(n_samples):
        # A. Structural Value
        p = base_price + (df.iloc[i]['surface_covered_in_m2'] * 2100)
        p += hood_premiums[df.iloc[i]['neighborhood']]
        
        # B. Coordinate Logic (THE FIX)
        # Instead of multiplying raw negative latitude, we calculate the "Delta"
        # Reference point: The southernmost point of the map (-34.70)
        # As you go North (Lat increases towards 0), value increases.
        current_lat = df.iloc[i]['lat']
        lat_delta = current_lat - (-34.70) # e.g., -34.60 - (-34.70) = +0.10 (Positive!)
        
        # Reference point: The westernmost point (-58.60)
        current_lon = df.iloc[i]['lon']
        lon_delta = current_lon - (-58.60)
        
        # Add value based on Location Delta (Positional Premium)
        p += (lat_delta * 500000) # Being 0.1 degree North adds $50k
        p += (lon_delta * 200000)
        
        # C. Market Volatility (Noise)
        market_variance = np.random.normal(0, 35000) 
        p += market_variance
        
        prices.append(p)
        
    y = np.array(prices)
    
    # 3. Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(df, y, test_size=0.2, random_state=42)
    
    # 4. Build Pipeline
    categorical_features = ['neighborhood']
    numerical_features = ['surface_covered_in_m2', 'lat', 'lon']
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features),
            ('num', 'passthrough', numerical_features)
        ]
    )
    
    model = make_pipeline(preprocessor, Ridge(alpha=1.0))
    model.fit(X_train, y_train)
    
    # 5. Evaluate
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    return model, mae, r2

model, mae, r2 = train_and_evaluate()

# --- DASH CONFIGURATION ---
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.LUX])
server = app.server

app.layout = dbc.Container([
    dbc.Row([
        dbc.Col(html.H1("Buenos Aires Real Estate Predictor", className="text-center text-primary mb-4"), width=12)
    ], className="mt-5"),

    dbc.Row([
        dbc.Col([
            dbc.Alert([
                html.H5("Model Validation Metrics (Test Set)", className="alert-heading"),
                html.P(f"R² Score: {r2:.2f}"), 
                html.P(f"Mean Absolute Error (MAE): ${mae:,.2f}"),
                html.Hr(),
                html.P("Metrics derived from unseen test split (20%) during initialization.", className="mb-0 small")
            ], color="info")
        ], width=12, md=12)
    ], className="mb-4"),

    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Property Parameters", className="fw-bold bg-primary text-white"),
                dbc.CardBody([
                    html.Label("Neighborhood"),
                    dcc.Dropdown(
                        id='neighborhood-input',
                        options=[
                            {'label': 'Puerto Madero', 'value': 'Puerto Madero'},
                            {'label': 'Palermo', 'value': 'Palermo'},
                            {'label': 'Recoleta', 'value': 'Recoleta'},
                            {'label': 'Belgrano', 'value': 'Belgrano'},
                            {'label': 'San Telmo', 'value': 'San Telmo'},
                            {'label': 'Caballito', 'value': 'Caballito'},
                            {'label': 'Boca', 'value': 'Boca'},
                        ],
                        value='Palermo', className="mb-3"
                    ),
                    html.Label("Surface Area (m²)"),
                    dcc.Slider(
                        id='area-input', min=30, max=250, step=5, value=60,
                        marks={i: f'{i}' for i in range(30, 251, 50)},
                        className="mb-4"
                    ),
                    # Updated Lat/Lon Inputs to match the simulation range
                    html.Label("Latitude"),
                    dbc.Input(id='lat-input', type='number', value=-34.58, step=0.001, className="mb-3"),
                    html.Label("Longitude"),
                    dbc.Input(id='lon-input', type='number', value=-58.40, step=0.001, className="mb-3"),
                ])
            ], className="shadow")
        ], width=12, md=4),

        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H4("Predicted Market Value", className="text-center text-muted mb-4"),
                    html.Div(id='prediction-output', className="display-4 text-center text-success fw-bold"),
                ])
            ], className="shadow h-100")
        ], width=12, md=8)
    ])
], fluid=True)

@app.callback(
    Output('prediction-output', 'children'),
    [Input('area-input', 'value'), Input('neighborhood-input', 'value'),
     Input('lat-input', 'value'), Input('lon-input', 'value')]
)
def update_prediction(area, neighborhood, lat, lon):
    input_df = pd.DataFrame({
        'surface_covered_in_m2': [float(area)],
        'lat': [float(lat)],
        'lon': [float(lon)],
        'neighborhood': [neighborhood]
    })
    try:
        pred = model.predict(input_df)[0]
        # Safety check: If price is somehow negative (extreme outlier inputs), cap it at 0
        if pred < 0: pred = 0
        return f"${pred:,.2f}"
    except Exception as e:
        return "Check Inputs"

if __name__ == '__main__':
    app.run_server(debug=True)
