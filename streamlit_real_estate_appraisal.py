# Streamlit Property Valuation App with Multi-Region Quantile Regression
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import shap
import warnings
import re
warnings.filterwarnings('ignore')

# ---------------- CONFIG ----------------
# Base directory of the current script
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"

# Multi-region configuration
REGION_CONFIG = {
    "BDF": {
        "name": "Bois-Des-Filion",
        "data_path": DATA_DIR / "donnees_BDF.csv",
        "feature_cols": ["Etage", "Age", "Aire_Batiment", "Aire_Lot", "Prox_Riverain"],
        "num_cols": ["Etage", "Age", "Aire_Batiment", "Aire_Lot"],
        "cat_cols": [],
        "model_prefix": "bdf"
    },
    "PMR": {
        "name": "Plateau Mont-Royal",
        "data_path": DATA_DIR / "Dataset_PMR.csv",
        "feature_cols": ["Category", "Etage", "Age", "Aire_Batiment", "Taxes_annuelles", "Near_A_Park", "Near_Metro_Station"],
        "num_cols": ["Etage", "Age", "Aire_Batiment", "Taxes_annuelles"],
        "cat_cols": ["Category", "Near_A_Park", "Near_Metro_Station"],
        "model_prefix": "pmr"
    },
    "Ste-Rose": {
        "name": "Sainte-Rose",
        "data_path": DATA_DIR / "Dataset_Ste-Rose.csv",
        "feature_cols": ["Etage", "Age", "Aire_Batiment", "Aire_Lot", "Prox_Riverain"],
        "num_cols": ["Etage", "Age", "Aire_Batiment", "Aire_Lot"],
        "cat_cols": [],
        "model_prefix": "ste_rose"
    }
}

# Create models directory if it doesn't exist
MODELS_DIR = BASE_DIR / "models"
MODELS_DIR.mkdir(exist_ok=True)

# --- Target handling ---
CANON_TARGET = "Prix_de_vente"

def load_region_dataframe_simple(region_key: str) -> pd.DataFrame:
    cfg = REGION_CONFIG[region_key]
    df = pd.read_csv(cfg["data_path"], encoding="utf-8-sig")
    df.columns = [c.strip() for c in df.columns]
    if CANON_TARGET not in df.columns:
        for alt in ["Prix_de_Vente", "prix_de_vente", "Prix", "Price", "price_sold"]:
            if alt in df.columns:
                df = df.rename(columns={alt: CANON_TARGET})
                break
    if CANON_TARGET not in df.columns:
        raise ValueError(f"[{cfg['name']}] Target '{CANON_TARGET}' not found. Columns: {list(df.columns)}")
    return df

def model_path_for(region_key, alpha):
    """Get model path for a specific region and quantile"""
    prefix = REGION_CONFIG[region_key]["model_prefix"]
    q = {0.05: "q05", 0.5: "q50", 0.95: "q95"}[alpha]
    return MODELS_DIR / f"{prefix}_model_{q}.joblib"

# Legacy compatibility - load BDF models if they exist
try:
    if (BASE_DIR / "property_model_q05.joblib").exists():
        # Migrate old models to new structure if needed
        pass
except:
    pass


# Page configuration
st.set_page_config(
    page_title="Property Valuation System - Quantile Regression",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .prediction-card {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 2rem;
        border-radius: 1rem;
        text-align: center;
        margin: 1rem 0;
    }
    .quantile-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 0.8rem;
        text-align: center;
        margin: 0.5rem 0;
    }
    .info-box {
        background-color: #e8f4fd;
        border: 1px solid #bee5eb;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 1rem 0;
    }
    .success-box {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 1rem 0;
    }
    .warning-box {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 1rem 0;
    }
    .stButton > button {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 0.5rem;
        padding: 0.5rem 2rem;
        font-weight: bold;
    }
    .stButton > button:hover {
        background: linear-gradient(90deg, #5a6fd8 0%, #6a4190 100%);
        transform: translateY(-2px);
        transition: all 0.3s ease;
    }
    /* Compact SHAP analysis styles with theme-consistent colors */
    .shap-card-grid {
        display: flex;
        gap: 0.5rem;
        flex-wrap: nowrap;
        overflow-x: auto;
        padding-bottom: 0.25rem;
        margin: 0.25rem 0 0.5rem 0;
    }
    .shap-card {
        border-radius: 0.5rem;
        padding: 0.5rem 0.75rem;
        min-width: 220px;
        box-shadow: none;
        border: 1px solid transparent;
        transition: all 0.2s ease;
    }
    .shap-card h5 {
        margin: 0 0 0.25rem 0;
        font-size: 0.9rem;
        color: #334155;
        font-weight: 600;
    }
    .shap-card .value {
        font-weight: 600;
        font-size: 1rem;
        margin: 0 0 0.25rem 0;
    }
    .shap-card .desc {
        font-size: 0.85rem;
        color: #4b5563;
        margin: 0;
    }
    
    /* Theme-consistent color palette for SHAP cards */
    .shap-card.building-efficiency {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
    }
    .shap-card.building-efficiency h5 { color: #f0f2f6; }
    .shap-card.building-efficiency .desc { color: #e8f4fd; }
    .shap-card.building-efficiency.positive .value { color: #ffffff; }
    .shap-card.building-efficiency.negative .value { color: #ffebee; }
    
    .shap-card.condition {
        background: linear-gradient(135deg, #1f77b4 0%, #4a90e2 100%);
        color: white;
    }
    .shap-card.condition h5 { color: #f0f2f6; }
    .shap-card.condition .desc { color: #e8f4fd; }
    .shap-card.condition.positive .value { color: #ffffff; }
    .shap-card.condition.negative .value { color: #ffebee; }
    
    .shap-card.premium-location {
        background: linear-gradient(135deg, #5a6fd8 0%, #8b5cf6 100%);
        color: white;
    }
    .shap-card.premium-location h5 { color: #f0f2f6; }
    .shap-card.premium-location .desc { color: #e8f4fd; }
    .shap-card.premium-location.positive .value { color: #ffffff; }
    .shap-card.premium-location.negative .value { color: #ffebee; }
    
    .shap-card.floor-level {
        background: linear-gradient(135deg, #64748b 0%, #94a3b8 100%);
        color: white;
    }
    .shap-card.floor-level h5 { color: #f0f2f6; }
    .shap-card.floor-level .desc { color: #e8f4fd; }
    .shap-card.floor-level.positive .value { color: #ffffff; }
    .shap-card.floor-level.negative .value { color: #ffebee; }
    
    .shap-card.lot-size {
        background: linear-gradient(135deg, #6b7280 0%, #9ca3af 100%);
        color: white;
    }
    .shap-card.lot-size h5 { color: #f0f2f6; }
    .shap-card.lot-size .desc { color: #e8f4fd; }
    .shap-card.lot-size.positive .value { color: #ffffff; }
    .shap-card.lot-size.negative .value { color: #ffebee; }
    
    .compact-info { margin: 0.25rem 0; font-size: 0.85rem; }
    .scroll-x { overflow-x: auto; }
</style>
""", unsafe_allow_html=True)

# ---------------- FONCTIONS ----------------
def _read_binary(path):
    with open(path, "rb") as f:
        return f.read()

def get_transformed_feature_names(preprocessor, num_cols, cat_cols):
    names = []
    for name, trans, cols in preprocessor.transformers_:
        if name == 'num':
            # passthrough numeric transformed by scaler; same count
            names.extend(list(cols))
        elif name == 'cat':
            ohe = trans.named_steps.get('onehot')
            if ohe is not None:
                ohe_names = ohe.get_feature_names_out(cols)
                names.extend(ohe_names.tolist())
            else:
                names.extend(list(cols))
    return names

def aggregate_shap_to_original(shap_vec, transformed_names, num_cols, cat_cols):
    """Sum SHAP contributions of one-hot expanded categories back into their original source feature names."""
    agg = {c: 0.0 for c in list(num_cols) + list(cat_cols)}
    for val, tname in zip(shap_vec, transformed_names):
        # tname examples: "onehot__CONDO_1" or "CONDO_1" depending on versions; use regex to recover base
        base = re.split(r'[_]{2}', tname)[-1]  # strip pipeline prefix if any
        base = re.split(r'_', base)[0]        # take the source col before first underscore
        if base in agg:
            agg[base] += float(val)
    return agg

def create_features(df, region_key="BDF", is_training=True):
    """Prepare raw data features for a specific region"""
    df = df.copy()
    
    # Get region configuration
    config = REGION_CONFIG[region_key]
    feature_cols = config["feature_cols"]
    num_cols = config["num_cols"]
    cat_cols = config["cat_cols"]
    
    # Auto-detect available columns and adapt feature_cols if needed
    available_cols = set(df.columns)
    adapted_feature_cols = [col for col in feature_cols if col in available_cols]
    
    if len(adapted_feature_cols) != len(feature_cols):
        st.warning(f"Some expected columns missing for {region_key}. Using available: {adapted_feature_cols}")
    
    # Ensure all required columns exist with sensible defaults
    for col in adapted_feature_cols:
        if col not in df.columns:
            if col in num_cols:
                df[col] = 0  # Default numeric value
            elif col in cat_cols:
                df[col] = "__missing__"  # Default categorical value
            else:
                df[col] = 0  # Default fallback
    
    # Handle PMR-specific boolean conversions
    if region_key == "PMR":
        # Convert boolean text columns to 0/1 if needed
        for col in ["Near_A_Park", "Near_Metro_Station"]:
            if col in df.columns:
                if df[col].dtype == 'object':
                    df[col] = df[col].map({'Yes': 1, 'No': 0, 'True': 1, 'False': 0, True: 1, False: 0}).fillna(0)
    
    if not is_training:
        df = df.fillna(0)
    
    # Return only the features needed by this region
    return df[adapted_feature_cols]

def train_quantile_models(region_key="BDF"):
    """Train quantile regression models for a specific region using sklearn Pipeline"""
    config = REGION_CONFIG[region_key]
    csv_path = config["data_path"]
    
    if not csv_path.exists():
        raise FileNotFoundError(f"Dataset introuvable : {csv_path}")
    
    with st.spinner(f"Loading and preparing data for {config['name']}..."):
        df = load_region_dataframe_simple(region_key)
        df = create_features(df, region_key, is_training=True)
        if CANON_TARGET not in df.columns:
            raise ValueError(f"[{config['name']}] Target '{CANON_TARGET}' missing after feature prep. Columns: {list(df.columns)}")
        df = df.dropna(subset=[CANON_TARGET])

        # Use ONLY the configured features that are present
        feature_cols = [c for c in config["feature_cols"] if c in df.columns]
        if not feature_cols:
            raise ValueError(f"[{config['name']}] No usable features. Columns: {list(df.columns)}")

        X = df[feature_cols]
        y = df[CANON_TARGET].astype(float)

    # Build preprocessing pipeline
    num_cols = [col for col in config["num_cols"] if col in feature_cols]
    cat_cols = [col for col in config["cat_cols"] if col in feature_cols]
    
    # Create transformers
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', RobustScaler())
    ])
    
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='__missing__')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', drop='first'))
    ])
    
    # Create preprocessor
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, num_cols),
            ('cat', categorical_transformer, cat_cols)
        ],
        remainder='drop'
    )

    metrics = {}
    
    with st.spinner("Training quantile models..."):
        for alpha in [0.05, 0.50, 0.95]:
            # Create full pipeline
            pipe = Pipeline([
                ('preproc', preprocessor),
                ('model', GradientBoostingRegressor(loss="quantile", alpha=alpha, random_state=42))
            ])
            
            # Train-test split
            X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # Train pipeline
            pipe.fit(X_train, y_train)
            
            # Store model data
            model_data = {
                "pipeline": pipe,
                "features": feature_cols,
                "region": region_key,
                "num_cols": num_cols,
                "cat_cols": cat_cols
            }
            
            # For the median model (alpha=0.50), also store MAE
            if alpha == 0.50:
                y_pred_val = pipe.predict(X_val)
                metrics["MAE"] = mean_absolute_error(y_val, y_pred_val)
                model_data["mae"] = metrics["MAE"]
            
            # Save pipeline
            path = model_path_for(region_key, alpha)
            joblib.dump(model_data, path)

    return metrics

def predict_with_models(region_key="BDF", **kwargs):
    """Make predictions using all three quantile models for a specific region"""
    # Prepare inputs based on region
    config = REGION_CONFIG[region_key]
    feature_cols = config["feature_cols"]
    
    # Create input DataFrame with all possible features
    inputs_dict = {}
    for col in feature_cols:
        if col in kwargs:
            inputs_dict[col] = kwargs[col]
        else:
            # Default values for missing features
            if col in config["num_cols"]:
                inputs_dict[col] = 0
            elif col in config["cat_cols"]:
                inputs_dict[col] = "__missing__"
            else:
                inputs_dict[col] = 0
    
    inputs = pd.DataFrame([inputs_dict])
    inputs = create_features(inputs, region_key, is_training=False)

    preds = {}
    for alpha in [0.05, 0.50, 0.95]:
        path = model_path_for(region_key, alpha)
        try:
            data = joblib.load(path)
            pipe = data["pipeline"]
            preds[alpha] = float(pipe.predict(inputs)[0])
        except FileNotFoundError:
            st.error(f"Model not found for {region_key} quantile {alpha}. Please train models first.")
            return None, None, None

    return preds[0.05], preds[0.50], preds[0.95]

def get_model_mae(region_key="BDF"):
    """Get the MAE from the trained model for a specific region"""
    try:
        # Load the median model to get the MAE
        path = model_path_for(region_key, 0.5)
        data = joblib.load(path)
        
        if "mae" in data:
            return data["mae"]
        else:
            # If MAE is not stored in the model, compute it from the training data
            df = load_region_dataframe_simple(region_key)
            df = create_features(df, region_key, is_training=True)
            df = df.dropna(subset=['Prix_de_vente'])
            
            feature_cols = data["features"]
            X = df[feature_cols]
            y = df['Prix_de_vente']
            
            pipe = data["pipeline"]
            y_pred = pipe.predict(X)
            
            return mean_absolute_error(y, y_pred)
    except Exception as e:
        st.warning(f"Could not retrieve MAE for {region_key}: {e}")
        return None

def create_shap_waterfall_chart(region_key="BDF", predicted_value=None, **kwargs):
    """
    Create a SHAP waterfall chart showing how the predicted price is constructed.
    """
    try:
        # Load the median model (q50)
        path = model_path_for(region_key, 0.5)
        data = joblib.load(path)
        pipe = data["pipeline"]
        features = data["features"]

        # Prepare single-row input as used by the model
        config = REGION_CONFIG[region_key]
        inputs_dict = {}
        for col in config["feature_cols"]:
            if col in kwargs:
                inputs_dict[col] = kwargs[col]
            else:
                if col in config["num_cols"]:
                    inputs_dict[col] = 0
                elif col in config["cat_cols"]:
                    inputs_dict[col] = "__missing__"
                else:
                    inputs_dict[col] = 0

        inputs = pd.DataFrame([inputs_dict])
        inputs = create_features(inputs, region_key, is_training=False)

        # Transform input and sample a background set in model's feature space
        X_transformed = pipe.named_steps["preproc"].transform(inputs)

        df_bg = load_region_dataframe_simple(region_key)
        df_bg = create_features(df_bg, region_key, is_training=True).dropna(subset=[CANON_TARGET])
        background_sample = df_bg[features].sample(min(100, len(df_bg)), random_state=42)
        X_background = pipe.named_steps["preproc"].transform(background_sample)

        # SHAP for GradientBoostingRegressor
        model = pipe.named_steps["model"]
        explainer = shap.TreeExplainer(model, data=X_background)
        shap_values = explainer.shap_values(X_transformed)  # shape: (1, n_features)
        base_value = float(explainer.expected_value)

        # Build waterfall data
        feature_names = features
        shap_contributions = shap_values[0]  # (n_features,)
        
        # Check for NaN values
        if np.isnan(shap_contributions).any():
            raise ValueError("NaN in SHAP contributions after preprocessing.")
        
        # Create waterfall data
        waterfall_data = []
        cumulative_value = base_value

        # Base
        waterfall_data.append({
            'feature': 'Base Value',
            'contribution': 0.0,
            'cumulative': base_value,
            'color': '#1f77b4'
        })

        # Contributions
        for feature, contribution in zip(feature_names, shap_contributions):
            c = float(contribution)
            cumulative_value += c
            color = '#2ca02c' if c >= 0 else '#d62728'
            waterfall_data.append({
                'feature': feature,
                'contribution': c,
                'cumulative': cumulative_value,
                'color': color
            })

        # Final prediction
        final_val = float(predicted_value) if predicted_value is not None else cumulative_value
        waterfall_data.append({
            'feature': 'Final Prediction',
            'contribution': 0.0,
            'cumulative': final_val,
            'color': '#1f77b4'
        })

        # Plotly figure
        fig = go.Figure()

        for i, point in enumerate(waterfall_data):
            if i == 0 or i == len(waterfall_data) - 1:
                # Base and Final as absolute bars
                cumulative_val = float(point['cumulative'])
                fig.add_trace(go.Bar(
                    x=[point['feature']],
                    y=[cumulative_val],
                    marker_color=point['color'],
                    name=point['feature'],
                    text=[f"${cumulative_val:,.0f}"],
                    textposition='auto',
                    showlegend=False
                ))
            else:
                # Contributions as delta bars
                contribution_val = float(point['contribution'])
                fig.add_trace(go.Bar(
                    x=[point['feature']],
                    y=[contribution_val],
                    marker_color=point['color'],
                    name=point['feature'],
                    text=[f"${contribution_val:,.0f}"],
                    textposition='auto',
                    showlegend=False
                ))

        # Flow line
        x_positions = list(range(len(waterfall_data)))
        cumulative_values = [float(pt['cumulative']) for pt in waterfall_data]
        fig.add_trace(go.Scatter(
            x=x_positions,
            y=cumulative_values,
            mode='lines+markers',
            line=dict(color='#666666', width=2, dash='dot'),
            marker=dict(size=6, color='#666666'),
            name='Flow',
            showlegend=False,
            hoverinfo='skip'
        ))

        fig.update_layout(
            title="Price Construction (SHAP Waterfall)",
            xaxis_title="Features",
            yaxis_title="Price ($)",
            height=500,
            showlegend=False,
            xaxis=dict(
                tickangle=45,
                tickmode='array',
                tickvals=list(range(len(waterfall_data))),
                ticktext=[pt['feature'] for pt in waterfall_data]
            ),
            yaxis=dict(tickformat='$,.0f'),
            margin=dict(b=100)
        )
        return fig

    except Exception as e:
        st.warning(f"Could not create SHAP waterfall chart: {e}")
        return None


def find_nearest_neighbors_and_calculate_ratio(region_key="BDF", predicted_value=None, k=3, **kwargs):
    """
    Find nearest neighbors in the dataset and calculate the ratio between predicted and actual prices.
    
    Args:
        region_key: Region identifier
        predicted_value: The predicted value from the model
        k: Number of nearest neighbors to consider (default=3)
        **kwargs: Property features
        
    Returns:
        dict: Contains ratio, actual_price, and whether averaging was used
    """
    try:
        # Load the dataset
        df = load_region_dataframe_simple(region_key)
        df = create_features(df, region_key, is_training=True)
        df = df.dropna(subset=[CANON_TARGET])
        
        # Small data safeguards
        n_samples = len(df)
        k = min(k, n_samples - 1)  # Ensure k doesn't exceed available samples
        
        # Load the pipeline for preprocessing
        path = model_path_for(region_key, 0.5)
        data = joblib.load(path)
        pipe = data["pipeline"]
        features = data["features"]
        
        # Prepare input data
        inputs_dict = {}
        for col in config["feature_cols"]:
            if col in kwargs:
                inputs_dict[col] = kwargs[col]
            else:
                if col in config["num_cols"]:
                    inputs_dict[col] = 0
                elif col in config["cat_cols"]:
                    inputs_dict[col] = "__missing__"
                else:
                    inputs_dict[col] = 0
        
        inputs = pd.DataFrame([inputs_dict])
        inputs = create_features(inputs, region_key, is_training=False)
        
        # Transform using pipeline preprocessor
        X_transformed = pipe.named_steps["preproc"].transform(inputs)
        
        # Transform all dataset features
        X_dataset = pipe.named_steps["preproc"].transform(df[features])
        
        # Check for exact match first in original feature space
        exact_match = df.copy()
        for col in config["feature_cols"]:
            if col in kwargs:
                exact_match = exact_match[exact_match[col] == kwargs[col]]
        
        if len(exact_match) > 0:
            # Use the first exact match
            actual_price = exact_match['Prix_de_vente'].iloc[0]
            used_averaging = False
        else:
            # Find k nearest neighbors using Euclidean distance in preprocessed space
            distances = np.sqrt(np.sum((X_dataset - X_transformed) ** 2, axis=1))
            
            # Get indices of k nearest neighbors
            nearest_indices = np.argsort(distances)[:k]
            
            # Get actual prices of nearest neighbors
            actual_prices = df.iloc[nearest_indices]['Prix_de_vente'].values
            
            # Use average of k nearest neighbors
            actual_price = np.mean(actual_prices)
            used_averaging = True
        
        # Calculate ratio
        ratio = (predicted_value / actual_price) * 100
        
        return {
            'ratio': ratio,
            'actual_price': actual_price,
            'used_averaging': used_averaging,
            'k': k
        }
        
    except Exception as e:
        st.warning(f"Could not calculate ratio for {region_key}: {e}")
        return None

def compute_shap_values(region_key="BDF", **kwargs):
    """
    Compute SHAP values for property prediction using the trained GradientBoostingRegressor model.
    Returns a dict keyed by original feature name.
    """
    try:
        path = model_path_for(region_key, 0.5)
        data = joblib.load(path)
        pipe = data["pipeline"]
        features = data["features"]

        # Build input row
        config = REGION_CONFIG[region_key]
        inputs_dict = {}
        for col in config["feature_cols"]:
            if col in kwargs:
                inputs_dict[col] = kwargs[col]
            else:
                if col in config["num_cols"]:
                    inputs_dict[col] = 0
                elif col in config["cat_cols"]:
                    inputs_dict[col] = "__missing__"
                else:
                    inputs_dict[col] = 0

        # Keep some raw values for explanations
        aire_batiment = float(inputs_dict.get("Aire_Batiment", 0) or 0)
        aire_lot = float(inputs_dict.get("Aire_Lot", 0) or 0)
        age = float(inputs_dict.get("Age", 0) or 0)
        etage = float(inputs_dict.get("Etage", 0) or 0)
        prox_riverain = int(inputs_dict.get("Prox_Riverain", 0) or 0)

        inputs = pd.DataFrame([inputs_dict])
        inputs = create_features(inputs, region_key, is_training=False)
        X_transformed = pipe.named_steps["preproc"].transform(inputs)

        # Background
        df = load_region_dataframe_simple(region_key)
        df = create_features(df, region_key, is_training=True).dropna(subset=[CANON_TARGET])
        background_sample = df[features].sample(min(100, len(df)), random_state=42)
        X_background = pipe.named_steps["preproc"].transform(background_sample)

        # SHAP
        model = pipe.named_steps["model"]
        explainer = shap.TreeExplainer(model, data=X_background)
        shap_values = explainer.shap_values(X_transformed)  # (1, n_features)

        shap_results = {}
        for i, feature in enumerate(features):
            sv = float(shap_values[0][i])

            if feature == "Aire_Batiment":
                per_m2 = (sv / aire_batiment) if aire_batiment > 0 else 0.0
                shap_results[feature] = {
                    "shap_value": sv,
                    "description": f"Each extra 10 m² adds ~${per_m2 * 10:,.0f}",
                    "label": "Building Efficiency"
                }

            elif feature == "Age":
                modern_threshold = 20
                if age < modern_threshold:
                    shap_results[feature] = {
                        "shap_value": sv,
                        "description": f"Modern building (Age < {modern_threshold}) adds ${sv:,.0f}",
                        "label": "Condition"
                    }
                else:
                    shap_results[feature] = {
                        "shap_value": sv,
                        "description": f"Older building (Age ≥ {modern_threshold}) reduces value by ${abs(sv):,.0f}",
                        "label": "Condition"
                    }

            elif feature == "Prox_Riverain":
                if prox_riverain == 1:
                    shap_results[feature] = {
                        "shap_value": sv,
                        "description": f"Waterfront location adds ${sv:,.0f}",
                        "label": "Premium Location"
                    }
                else:
                    shap_results[feature] = {
                        "shap_value": sv,
                        "description": f"Non-waterfront location: ${sv:,.0f} impact",
                        "label": "Premium Location"
                    }

            elif feature == "Etage":
                if etage > 1:
                    shap_results[feature] = {
                        "shap_value": sv,
                        "description": f"Multi-floor ({int(etage)} floors) adds ${sv:,.0f}",
                        "label": "Floor Level"
                    }
                else:
                    shap_results[feature] = {
                        "shap_value": sv,
                        "description": f"Single floor: ${sv:,.0f} impact",
                        "label": "Floor Level"
                    }

            elif feature == "Aire_Lot":
                per_m2 = (sv / aire_lot) if aire_lot > 0 else 0.0
                shap_results[feature] = {
                    "shap_value": sv,
                    "description": f"Each extra 10 m² adds ~${per_m2 * 10:,.0f}",
                    "label": "Lot Size"
                }

            # For other features (e.g., PMR category/park/metro), you can add cases later.
        return shap_results

    except Exception as e:
        st.warning(f"Could not compute SHAP values: {e}")
        return None




def main():
    # Header
    st.markdown('<h1 class="main-header">🏠 Property Valuation System</h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; font-size: 1.2rem; color: #666;">Multi-Region Gradient Boosting with Quantile Loss for Accurate Property Valuations</p>', unsafe_allow_html=True)
    
    # Region selector
    st.sidebar.title("Region Selection")
    region_key = st.sidebar.selectbox(
        "Select Region:",
        ["BDF", "PMR", "Ste-Rose"],
        index=0,
        format_func=lambda x: REGION_CONFIG[x]["name"]
    )
    
    # Show selected region info
    selected_region = REGION_CONFIG[region_key]
    st.sidebar.info(f"**Selected:** {selected_region['name']}")
    
    # Debug info checkbox
    show_debug = st.sidebar.checkbox("Show debug info", value=False)
    
    # Sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox(
        "Choose a page:",
        ["Home", "Model Performance", "Property Valuation", "About"]
    )
    
    if page == "Home":
        st.markdown("""
        ## Welcome to the Advanced Property Valuation System
        
        This system uses **Gradient Boosting with Quantile Loss** to provide not just point estimates, but also confidence intervals for property valuations.
        
        ### Key Features:
        - **Gradient Boosting with Quantile Loss**: Provides 5th, 50th, and 95th percentile predictions
        - **Confidence Intervals**: Understand the uncertainty in your valuations
        - **Advanced Features**: Raw attributes for more relevant insights
        
        ### How to Use:
        1. **Train the Model**: Go to "Model Performance" to train the quantile models
        2. **Get Valuations**: Use "Property Valuation" for custom estimates
        """)
        
        # Check model status
        model_exists = model_path_for(region_key, 0.5).exists()
        if model_exists:
            st.markdown('<div class="success-box">', unsafe_allow_html=True)
            st.markdown("✅ **Model Status**: Quantile models are trained and ready to use!")
            st.markdown("</div>", unsafe_allow_html=True)
        else:
            st.markdown('<div class="warning-box">', unsafe_allow_html=True)
            st.markdown("⚠️ **Model Status**: Models need to be trained for this region. Go to 'Model Performance' to train the models.")
            st.markdown("</div>", unsafe_allow_html=True)
        
        # Quick stats
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Model Type", "Gradient Boosting with Quantile Loss", "Advanced")
        
        with col2:
            st.metric("Prediction Range", "5th - 95th percentile", "Confidence Intervals")
        
        with col3:
            st.metric("Features", "5 Raw Attributes", "Raw attributes for more relevant insights")
    
    elif page == "Property Valuation":
        st.header(f"🏠 Property Valuation - {selected_region['name']}")
        
        # Check if models exist for this region
        model_path = model_path_for(region_key, 0.5)
        if not model_path.exists():
            st.error(f"⚠️ Models for {selected_region['name']} need to be trained first. Please go to 'Model Performance' to train the models.")
            return
        
        # Dynamic input fields based on region
        with st.form("valuation_form"):
            st.subheader("Enter Property Details")
            
            # Get region configuration
            config = REGION_CONFIG[region_key]
            feature_cols = config["feature_cols"]
            num_cols = config["num_cols"]
            cat_cols = config["cat_cols"]
            
            # Create input fields dynamically
            input_values = {}
            
            # Split into two columns
            col1, col2 = st.columns(2)
            
            with col1:
                for i, feature in enumerate(feature_cols[:len(feature_cols)//2 + 1]):
                    if feature == "Etage":
                        input_values[feature] = st.number_input("Floor", min_value=1, max_value=20, value=2)
                    elif feature == "Age":
                        input_values[feature] = st.number_input("Building Age (years)", min_value=0, max_value=100, value=15)
                    elif feature == "Aire_Batiment":
                        input_values[feature] = st.number_input("Building Area (m²)", min_value=20.0, max_value=1000.0, value=120.0, step=10.0)
                    elif feature == "Aire_Lot":
                        input_values[feature] = st.number_input("Lot Area (m²)", min_value=50.0, max_value=2000.0, value=300.0, step=50.0)
                    elif feature == "Prox_Riverain":
                        input_values[feature] = st.selectbox("Waterfront Proximity", options=[0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
                    elif feature == "Taxes_annuelles":
                        input_values[feature] = st.number_input("Annual Taxes ($)", min_value=0.0, max_value=50000.0, value=3000.0, step=100.0)
                    elif feature == "Category":
                        # Load unique categories from PMR dataset
                        try:
                            df_pmr = load_region_dataframe_simple(region_key)
                            categories = df_pmr["Category"].unique().tolist()
                            input_values[feature] = st.selectbox("Category", options=categories)
                        except:
                            input_values[feature] = st.text_input("Category", value="Residential")
                    elif feature in ["Near_A_Park", "Near_Metro_Station"]:
                        input_values[feature] = st.selectbox(feature.replace("_", " "), options=[0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
            
            with col2:
                for feature in feature_cols[len(feature_cols)//2 + 1:]:
                    if feature == "Etage":
                        input_values[feature] = st.number_input("Floor", min_value=1, max_value=20, value=2)
                    elif feature == "Age":
                        input_values[feature] = st.number_input("Building Age (years)", min_value=0, max_value=100, value=15)
                    elif feature == "Aire_Batiment":
                        input_values[feature] = st.number_input("Building Area (m²)", min_value=20.0, max_value=1000.0, value=120.0, step=10.0)
                    elif feature == "Aire_Lot":
                        input_values[feature] = st.number_input("Lot Area (m²)", min_value=50.0, max_value=2000.0, value=300.0, step=50.0)
                    elif feature == "Prox_Riverain":
                        input_values[feature] = st.selectbox("Waterfront Proximity", options=[0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
                    elif feature == "Taxes_annuelles":
                        input_values[feature] = st.number_input("Annual Taxes ($)", min_value=0.0, max_value=50000.0, value=3000.0, step=100.0)
                    elif feature == "Category":
                        try:
                            df_pmr = load_region_dataframe_simple(region_key)
                            categories = df_pmr["Category"].unique().tolist()
                            input_values[feature] = st.selectbox("Category", options=categories)
                        except:
                            input_values[feature] = st.text_input("Category", value="Residential")
                    elif feature in ["Near_A_Park", "Near_Metro_Station"]:
                        input_values[feature] = st.selectbox(feature.replace("_", " "), options=[0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
            
            submitted = st.form_submit_button("Get Valuation")
            
            if submitted:
                try:
                    # Validate inputs
                    if "Aire_Batiment" in input_values and input_values["Aire_Batiment"] <= 0:
                        st.error("Please enter a valid building area.")
                        return
                    
                    # Make prediction with region-aware function
                    low, median, high = predict_with_models(region_key, **input_values)
                    
                    if low is None or median is None or high is None:
                        st.error("Prediction failed. Please check if models are trained.")
                        return
                    
                    # Get MAE for display
                    mae = get_model_mae(region_key)
                    
                    # Calculate ratio for KPI
                    ratio_result = find_nearest_neighbors_and_calculate_ratio(
                        region_key=region_key, predicted_value=median, **input_values
                    )
                    
                    # Display results
                    st.markdown('<div class="prediction-card">', unsafe_allow_html=True)
                    st.markdown(f"## Estimated Property Value")
                    st.markdown(f"# ${median:,.0f}")
                    if mae is not None:
                        st.markdown(f"*MAE: ${mae:,.0f}*")
                    st.markdown("</div>", unsafe_allow_html=True)
                    
                    # Add explanation about the model
                    st.info("This model uses Gradient Boosting with Quantile Loss, a machine learning method that provides price predictions along with confidence ranges.")
                    
                    # Quantile breakdown
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.markdown('<div class="quantile-card">', unsafe_allow_html=True)
                        st.markdown("### Lower Bound")
                        st.markdown(f"## $ {low:,.0f}")
                        st.markdown("*5th percentile*")
                        st.markdown("</div>", unsafe_allow_html=True)
                    
                    with col2:
                        st.markdown('<div class="quantile-card">', unsafe_allow_html=True)
                        st.markdown("### Median")
                        st.markdown(f"## $ {median:,.0f}")
                        st.markdown("*50th percentile*")
                        st.markdown("</div>", unsafe_allow_html=True)
                    
                    with col3:
                        st.markdown('<div class="quantile-card">', unsafe_allow_html=True)
                        st.markdown("### Upper Bound")
                        st.markdown(f"## $ {high:,.0f}")
                        st.markdown("*95th percentile*")
                        st.markdown("</div>", unsafe_allow_html=True)
                    
                    # SHAP Waterfall Chart
                    st.subheader("Price Construction (SHAP Waterfall)")
                    fig = create_shap_waterfall_chart(region_key=region_key, predicted_value=median, **input_values)
                    if fig:
                        st.plotly_chart(fig, use_container_width=True)
                        st.info("💡 **Price Construction**: This chart starts from the dataset's average property value and shows how each attribute contributes positively or negatively to the final predicted price.")
                    else:
                        st.warning("Could not generate the price construction chart.")
                    
                    # Additional metrics
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        price_per_m2 = median / float(input_values.get("Aire_Batiment", 0) or 1)
                        st.metric("Price per m²", f"${price_per_m2:,.0f}")
                    
                    with col2:
                        if ratio_result:
                            st.metric("Ratio Valuation vs Price", f"{ratio_result['ratio']:.2f}%")
                            st.caption(f"Actual Price: ${ratio_result['actual_price']:,.0f}")
                            if ratio_result['used_averaging']:
                                st.caption(f"Average of {ratio_result['k']} similar properties")
                        else:
                            st.metric("Ratio Valuation vs Price", "N/A")
                    
                    with col3:
                        if mae is not None:
                            st.metric("Mean Absolute Error", f"${mae:,.0f}", "Average prediction error")
                        else:
                            st.metric("Mean Absolute Error", "N/A", "Not available")
                    
                    # Add helper text for the ratio KPI
                    if ratio_result:
                        st.info("💡 **Ratio Valuation vs Price**: This ratio compares the model's valuation to the closest known property or an average of similar properties in the dataset. 100% = perfect match, >100% = overestimation, <100% = underestimation.")

                    
                    # SHAP-based Property Analysis
                    st.subheader("Property Analysis (SHAP values)")
                    
                    # Compute SHAP values for this specific property
                    # SHAP values explain how each feature contributes to the prediction in dollars
                    # Positive values increase the predicted price, negative values decrease it
                    shap_results = compute_shap_values(region_key=region_key, **input_values)
                    
                    if shap_results:
                        st.markdown('<div class="compact-info">💡 <strong>SHAP Analysis</strong>: Each value shows how much each feature contributes to the predicted price in dollars.</div>', unsafe_allow_html=True)
                        
                        # Maintain logical grouping but render as compact horizontal cards
                        shap_sections = {
                            "Building Efficiency": ["Aire_Batiment"],
                            "Condition": ["Age"],
                            "Premium Location": ["Prox_Riverain"],
                            "Floor Level": ["Etage"],
                            "Lot Size": ["Aire_Lot"]
                        }
                        
                        cards_html_parts = []
                        for section_name, features in shap_sections.items():
                            for feature in features:
                                if feature in shap_results:
                                    result = shap_results[feature]
                                    shap_val = float(result["shap_value"]) if hasattr(result["shap_value"], "__float__") else result["shap_value"]
                                    desc = result["description"]
                                    label = result.get("label", feature)
                                    is_positive = shap_val >= 0
                                    icon = "📈" if is_positive else "📉"
                                    sign_class = "positive" if is_positive else "negative"
                                    
                                    # Map section names to theme-consistent CSS classes
                                    section_class_map = {
                                        "Building Efficiency": "building-efficiency",
                                        "Condition": "condition", 
                                        "Premium Location": "premium-location",
                                        "Floor Level": "floor-level",
                                        "Lot Size": "lot-size"
                                    }
                                    theme_class = section_class_map.get(section_name, "building-efficiency")
                                    
                                    cards_html_parts.append(
                                        f"<div class='shap-card {theme_class} {sign_class}'>"
                                        f"<h5>{label}</h5>"
                                        f"<div class='value'>{icon} ${shap_val:,.0f}</div>"
                                        f"<p class='desc'>{desc}</p>"
                                        f"</div>"
                                    )

                        if cards_html_parts:
                            cards_html = "".join(cards_html_parts)
                            st.markdown(f"<div class='shap-card-grid'>{cards_html}</div>", unsafe_allow_html=True)
                    else:
                        st.warning("SHAP analysis could not be computed for this property.")
                    
                except Exception as e:
                    st.error(f"An error occurred: {e}")
    

    
    elif page == "Model Performance":
        st.header(f"📈 Model Training & Performance - {selected_region['name']}")
        
        # Check if models exist for this region
        model_path = model_path_for(region_key, 0.5)
        
        if st.button(f"🚀 Train Quantile Models for {selected_region['name']}", type="primary"):
            try:
                with st.spinner(f"Training models for {selected_region['name']}..."):
                    metrics = train_quantile_models(region_key)
                
                st.success(f"✅ Models for {selected_region['name']} trained and saved successfully!")
                
                # Display metrics
                if "MAE" in metrics:
                    st.metric("Mean Absolute Error", f"${metrics['MAE']:,.0f}")
                    st.markdown("*Average prediction error*")
                
                if "R2" in metrics:
                    st.metric("R² Score", f"{metrics['R2']:.3f}")
                    st.markdown("*Model fit quality*")
                
                # Temporary Model Downloads
                st.subheader("Temporary Model Downloads (this branch only)")
                alpha_map = {0.05: "q05", 0.50: "q50", 0.95: "q95"}
                model_paths = [model_path_for(region_key, a) for a in (0.05, 0.50, 0.95)]
                alphas = [0.05, 0.50, 0.95]
                
                for alpha, path in zip(alphas, model_paths):
                    if path.exists():
                        st.download_button(
                            label=f"⬇️ Download model ({alpha_map[alpha]})",
                            data=_read_binary(path),
                            file_name=path.name,
                            mime="application/octet-stream"
                        )
                st.caption("These download buttons are temporary and will be removed when merging to main.")
                
                # Model info
                st.subheader("Model Information")
                config = REGION_CONFIG[region_key]
                st.markdown(f"""
                **Gradient Boosting with Quantile Loss Models Trained for {selected_region['name']}:**
                - **5th percentile model**: Lower bound predictions
                - **50th percentile model**: Median predictions  
                - **95th percentile model**: Upper bound predictions
                
                **Features Used:**
                - {', '.join(config['feature_cols'])}
                - Raw property attributes from the dataset
                - Full sklearn Pipeline with preprocessing
                """)
                
                if show_debug:
                    st.info(f"**Debug Info:** Models saved to {MODELS_DIR}")
                
            except Exception as e:
                st.error(f"❌ Training failed for {selected_region['name']}: {e}")
        
        elif model_path.exists():
            st.success(f"✅ Models for {selected_region['name']} are already trained and ready to use!")
            
            # Load and display model info
            try:
                data = joblib.load(model_path)
                st.info("**Model Details:**")
                st.write(f"- Features: {len(data['features'])} attributes")
                st.write(f"- Region: {data.get('region', 'Unknown')}")
                st.write("- Algorithm: Gradient Boosting with Quantile Loss")
                st.write("- Pipeline: Full sklearn Pipeline with preprocessing")
                
                if "mae" in data:
                    st.write(f"- MAE: ${data['mae']:,.0f}")
                
                if show_debug:
                    st.info(f"**Debug Info:** Model loaded from {model_path}")
                    st.write(f"Pipeline steps: {list(data['pipeline'].named_steps.keys())}")
                    
            except Exception as e:
                st.warning(f"Model files exist but may be corrupted: {e}. Please retrain.")
        else:
            st.info(f"ℹ️ No models found for {selected_region['name']}. Click the button above to train models.")
    
    elif page == "About":
        st.header("ℹ️ About")
        # Note: Using "Gradient Boosting with Quantile Loss" instead of "Quantile Regression" 
        # because quantile regression typically refers to linear methods, while this app uses GradientBoostingRegressor with quantile loss
        
        st.markdown("""
        ## Advanced Property Valuation System
        
        This system uses **Gradient Boosting with Quantile Loss** to provide comprehensive property valuations with confidence intervals.
        
        ### Technology Stack:
        - **Python**: Core programming language
        - **Streamlit**: Web application framework
        - **Scikit-learn**: Machine learning library with Gradient Boosting and quantile loss
        - **Plotly**: Interactive visualizations
        - **Pandas & NumPy**: Data manipulation
        
        ### Model Architecture:
        The system uses **Gradient Boosting with Quantile Loss**:
        - **5th percentile model**: Lower bound predictions (conservative estimate)
        - **50th percentile model**: Median predictions (most likely value)
        - **95th percentile model**: Upper bound predictions (optimistic estimate)
        
        ### Features Used:
        - Floor level
        - Building age (years)
        - Building area (m²)
        - Lot area (m²)
        - Waterfront proximity (Yes/No)
        - Raw property attributes from the dataset
        
        ### Advantages of Gradient Boosting with Quantile Loss:
        - **Uncertainty Quantification**: Provides confidence intervals
        - **Robust Predictions**: Less sensitive to outliers
        - **Risk Assessment**: Helps understand prediction uncertainty
        - **Better Decision Making**: Range of possible values instead of single point
        
        ### Data Source:
        The model is trained on real estate data from the Land Register of Quebec.
        
        ### Disclaimer
        
        This application does not replace the advice of a certified appraiser. However, it uses machine learning to predict the price of a property based on real property data. Statistical validity may vary from one model to another depending on the ML algorithm used.
        
        ---
        **Note**:  
        The model is evaluated internally using several statistical metrics, but these are kept under the hood.  
        What you see in the app are the confidence intervals and error metrics (such as MAE), which are the most relevant for decision-making.
        """)

if __name__ == "__main__":
    main()
