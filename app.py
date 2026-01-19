import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import dash_bootstrap_components as dbc
import pandas as pd
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import Ridge
# We use the standard sklearn OneHotEncoder for simplicity in deployment
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

# --- 1. REAL MODEL TRAINING (ON STARTUP) ---
# Since we can't upload the huge CSV, we generate realistic synthetic data
# and train a REAL Ridge Regression model on the fly.

def train_real_model():
    np.random.seed(42)
    n_samples = 500
    
    # Generate random features
    areas = np.random.randint(30, 200, n_samples)
    lats = np.random.uniform(-34.65, -34.55, n_samples)
    lons = np.random.uniform(-58.50, -58.35, n_samples)
    neighborhoods = np.random.choice(
        ['Palermo', 'Recoleta', 'Belgrano', 'San Telmo', 'Caballito', 'Boca', 'Puerto Madero'], 
        n_samples
    )
    
    # Create DataFrame
    df_train = pd.DataFrame({
        'surface_covered_in_m2': areas,
        'lat': lats,
        'lon': lons,
        'neighborhood': neighborhoods
    })
    
    # Define "True" weights for the target variable (Price)
    # This ensures our model actually learns relationships that make sense
    base_price = 10000
    
    # pricing logic: 
    # - Area adds value ($2000/m2)
    # - Latitude: moving North (increasing lat) in BA is generally pricier
    # - Neighborhood premiums
    hood_premiums = {
        'Puerto Madero': 100000, 'Palermo': 50000, 'Recoleta': 45000,
        'Belgrano': 40000, 'San Telmo': 20000, 'Caballito': 25000, 'Boca': 5000
    }
    
    prices = []
    for i in range(n_samples):
        p = base_price + (df_train.iloc[i]['surface_covered_in_m2'] * 2000)
        p += hood_premiums[df_train.iloc[i]['neighborhood']]
        # Add Lat/Lon effect (Simplified: Closer to specific point = more expensive)
        # Just adding noise and lat effect for the regression to find
        p += (df_train.iloc[i]['lat'] * 10000) 
        # Add random noise
        p += np.random.normal(0, 5000)
        prices.append(p)
        
    y_train = np.array(prices)
    
    # --- BUILD PIPELINE ---
    # We use ColumnTransformer to OneHotEncode ONLY the neighborhood
    # and pass the numericals (area, lat, lon) through.
    
    categorical_features = ['neighborhood']
    numerical_features = ['surface_covered_in_m2', 'lat', 'lon']
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features),
            ('num', 'passthrough', numerical_features)
        ]
    )
    
    model = make_pipeline(
        preprocessor,
        Ridge(alpha=1.0)
    )
    
    # Train the model
    model.fit(df_train, y_train)
    return model

# Train the model once when app starts
model = train_real_model()

# --- 2. DASH APP SETUP ---
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.LUX])
server = app.server

# --- 3. LAYOUT ---
app.layout = dbc.Container([
    dbc.Row([
        dbc.Col(html.H1("Buenos Aires Real Estate Predictor", className="text-center text-primary mb-4"), width=12)
    ], className="mt-5"),

    dbc.Row([
        # Left Column: Inputs
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Property Parameters", className="fw-bold bg-primary text-white"),
                dbc.CardBody([
                    
                    # Neighborhood
                    html.Label("Neighborhood", className="fw-bold"),
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
                        value='Palermo',
                        className="mb-3"
                    ),

                    # Area
                    html.Label("Surface Area (m²)", className="fw-bold"),
                    dcc.Slider(
                        id='area-input',
                        min=30, max=200, step=5, value=60,
                        marks={i: f'{i}' for i in range(30, 201, 40)},
                        tooltip={"placement": "bottom", "always_visible": True},
                        className="mb-4"
                    ),

                    # Latitude
                    html.Label("Latitude (-34.55 to -34.70)", className="fw-bold"),
                    dbc.Input(
                        id='lat-input', type='number', 
                        value=-34.58, step=0.001, 
                        className="mb-3"
                    ),

                    # Longitude
                    html.Label("Longitude (-58.35 to -58.55)", className="fw-bold"),
                    dbc.Input(
                        id='lon-input', type='number', 
                        value=-58.40, step=0.001, 
                        className="mb-3"
                    ),
                ])
            ], className="shadow")
        ], width=12, md=4),

        # Right Column: Prediction & Info
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H4("Predicted Market Value", className="text-center text-muted mb-4"),
                    html.Div(id='prediction-output', className="display-4 text-center text-success fw-bold"),
                    html.Hr(),
                    html.P([
                        "Model: ", html.Span("Ridge Regression (L2 Regularization)", className="fw-bold")
                    ], className="text-center"),
                    html.P([
                        "Pipeline: ", html.Span("OneHotEncoder + Numerical Passthrough", className="fw-bold")
                    ], className="text-center"),
                ])
            ], className="shadow h-100")
        ], width=12, md=8)
    ])
], fluid=True)

# --- 4. CALLBACK ---
@app.callback(
    Output('prediction-output', 'children'),
    [Input('area-input', 'value'),
     Input('neighborhood-input', 'value'),
     Input('lat-input', 'value'),
     Input('lon-input', 'value')]
)
def update_prediction(area, neighborhood, lat, lon):
    # Construct input dataframe exactly as the model expects
    input_df = pd.DataFrame({
        'surface_covered_in_m2': [float(area)],
        'lat': [float(lat)],
        'lon': [float(lon)],
        'neighborhood': [neighborhood]
    })
    
    try:
        pred = model.predict(input_df)[0]
        return f"${pred:,.2f}"
    except Exception as e:
        return "Check Inputs"

if __name__ == '__main__':
    app.run_server(debug=True)
