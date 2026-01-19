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

# --- 1. REAL MODEL TRAINING & METRICS ---
def train_and_evaluate():
    np.random.seed(42)
    n_samples = 1000  # Increased sample size for better stats
    
    # Generate Synthetic Data
    areas = np.random.randint(30, 200, n_samples)
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
    
    # Generate Target (Price) with some logic + noise
    base_price = 10000
    hood_premiums = {
        'Puerto Madero': 100000, 'Palermo': 50000, 'Recoleta': 45000,
        'Belgrano': 40000, 'San Telmo': 20000, 'Caballito': 25000, 'Boca': 5000
    }
    
    prices = []
    for i in range(n_samples):
        p = base_price + (df.iloc[i]['surface_covered_in_m2'] * 2000)
        p += hood_premiums[df.iloc[i]['neighborhood']]
        p += (df.iloc[i]['lat'] * 10000) 
        p += np.random.normal(0, 5000) # Noise
        prices.append(p)
        
    y = np.array(prices)
    
    # Split Data (80% Train, 20% Test)
    X_train, X_test, y_train, y_test = train_test_split(df, y, test_size=0.2, random_state=42)
    
    # Build Pipeline
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
    
    # Calculate Metrics
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    return model, mae, r2

# Train model and get metrics
model, mae, r2 = train_and_evaluate()

# --- 2. DASH APP SETUP ---
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.LUX])
server = app.server

# --- 3. LAYOUT ---
app.layout = dbc.Container([
    dbc.Row([
        dbc.Col(html.H1("Buenos Aires Real Estate Predictor", className="text-center text-primary mb-4"), width=12)
    ], className="mt-5"),

    # --- NEW: METRICS CARD ---
    dbc.Row([
        dbc.Col([
            dbc.Alert([
                html.H5("Model Performance (Test Set)", className="alert-heading"),
                html.P(f"R² Score: {r2:.2f} (Explains {r2*100:.0f}% of variance)"),
                html.P(f"Mean Absolute Error (MAE): ${mae:,.2f}"),
                html.Hr(),
                html.P("These metrics are calculated live on unseen test data during deployment.", className="mb-0 small")
            ], color="info")
        ], width=12, md=12)
    ], className="mb-4"),

    dbc.Row([
        # Inputs
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
                        id='area-input', min=30, max=200, step=5, value=60,
                        marks={i: f'{i}' for i in range(30, 201, 40)},
                        className="mb-4"
                    ),
                    html.Label("Latitude"),
                    dbc.Input(id='lat-input', type='number', value=-34.58, step=0.001, className="mb-3"),
                    html.Label("Longitude"),
                    dbc.Input(id='lon-input', type='number', value=-58.40, step=0.001, className="mb-3"),
                ])
            ], className="shadow")
        ], width=12, md=4),

        # Prediction
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H4("Predicted Value", className="text-center text-muted mb-4"),
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
        return f"${pred:,.2f}"
    except:
        return "Check Inputs"

if __name__ == '__main__':
    app.run_server(debug=True)
