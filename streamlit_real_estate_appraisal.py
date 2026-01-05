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
        "num_cols": ["Etage", "Age", "Aire_Batiment", "Aire_Lot", "Prox_Riverain"],
        "cat_cols": [],
        "model_prefix": "bdf"
    },
    "PMR": {
        "name": "Plateau Mont-Royal",
        "data_path": DATA_DIR / "Dataset_PMR.csv",
        "feature_cols": ["CONDO", "5PLEX_ET_MOINS", "6PLEX_ET_PLUS", "UNIFAMILIALE", "ETAGES", "AGE", "AIRE_HABITABLE", "TAXES_AN", "Prox_Parc", "Prox_Metro"],
        "num_cols": ["CONDO", "5PLEX_ET_MOINS", "6PLEX_ET_PLUS", "UNIFAMILIALE", "ETAGES", "AGE", "AIRE_HABITABLE", "TAXES_AN", "Prox_Parc", "Prox_Metro"],
        "cat_cols": [],
        "model_prefix": "pmr"
    },
    "Ste-Rose": {
        "name": "Sainte-Rose",
        "data_path": DATA_DIR / "Dataset_Ste-Rose.csv",
        "feature_cols": ["Etage", "Age", "Aire_Batiment_m2", "Aire_Lot_m2", "Garage", "Amenagement_paysager"],
        "num_cols": ["Etage", "Age", "Aire_Batiment_m2", "Aire_Lot_m2", "Garage", "Amenagement_paysager"],
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

def pmr_submodel_path_for(submodel_type, alpha):
    """Get model path for PMR submodel (CONDO or PLEX_SFH) and quantile"""
    q = {0.05: "Q05", 0.5: "Q50", 0.95: "Q95"}[alpha]
    return MODELS_DIR / f"PMR_{submodel_type}_{q}.joblib"

def models_available(region_key: str):
    """Check if all three quantile models exist for a region"""
    if region_key == "PMR":
        # For PMR, check both submodels exist
        return (all(pmr_submodel_path_for("CONDO", a).exists() for a in (0.05, 0.5, 0.95)) and
                all(pmr_submodel_path_for("PLEX_SFH", a).exists() for a in (0.05, 0.5, 0.95)))
    else:
        return all(model_path_for(region_key, a).exists() for a in (0.05, 0.5, 0.95))

def build_input_form(region_key: str):
    """Build input form for a specific region and return inputs, submitted flag, and missing fields"""
    cfg = REGION_CONFIG[region_key]
    num_cols = [c for c in cfg["num_cols"] if c in cfg["feature_cols"]]
    cat_cols = [c for c in cfg["cat_cols"] if c in cfg["feature_cols"]]
    inputs = {}

    with st.form("valuation_inputs", clear_on_submit=False):
        st.subheader("Enter property attributes")
        
        # Region-specific input handling
        if region_key == "BDF":
            # BDF: Remove Prox_Riverain from numeric inputs (will be handled as Yes/No)
            num_cols = [c for c in num_cols if c != "Prox_Riverain"]
            
            # Label mapping for BDF
            label_map = {
                "Etage": "Floor(s)",
                "Age": "Age",
                "Aire_Batiment": "Building Area (m2)",
                "Aire_Lot": "Lot Area (m2)"
            }
            
            # Numeric inputs with renamed labels
            for col in num_cols:
                label = label_map.get(col, col)
                step = 10.0 if any(k in col.lower() for k in ["aire", "m2", "lot"]) else 1.0
                minv = 0.0
                val = st.number_input(label, min_value=float(minv), step=float(step), key=f"bdf_{col}")
                inputs[col] = val
            
            # Waterfront Proximity as Yes/No (single input, no duplicate)
            prox_riverain_choice = st.selectbox("Waterfront Proximity", ["No", "Yes"], key="bdf_prox_riverain")
            inputs["Prox_Riverain"] = 1 if prox_riverain_choice == "Yes" else 0
        
        elif region_key == "PMR":
            # PMR: Property Type with UI-friendly labels mapped to internal values
            property_type_ui = st.selectbox(
                "Property Type",
                ["Condo", "Plex", "SFH"],
                help="Select the property type",
                key="pmr_property_type"
            )
            # Map UI values to internal model values
            property_type_map = {
                "Condo": "CONDO",
                "Plex": "5PLEX_ET_MOINS",  # Map Plex to 5PLEX_ET_MOINS
                "SFH": "UNIFAMILIALE"
            }
            inputs["Property_Type"] = property_type_map[property_type_ui]
            
            # Exclude property type columns and TAXES_AN from numeric inputs
            exclude_cols = ["CONDO", "5PLEX_ET_MOINS", "6PLEX_ET_PLUS", "UNIFAMILIALE", "TAXES_AN", "Prox_Parc", "Prox_Metro"]
            num_cols = [c for c in num_cols if c not in exclude_cols]
            
            # Label mapping for PMR numeric inputs
            label_map = {
                "ETAGES": "Floor(s)",
                "AGE": "Age",
                "AIRE_HABITABLE": "Building Area (m2)"
            }
            
            # Numeric inputs with renamed labels
            for col in num_cols:
                label = label_map.get(col, col)
                step = 10.0 if any(k in col.lower() for k in ["aire", "m2", "lot"]) else 1.0
                minv = 0.0
                val = st.number_input(label, min_value=float(minv), step=float(step), key=f"pmr_{col}")
                inputs[col] = val
            
            # Park Proximity as Yes/No (single input, no duplicate)
            prox_parc_choice = st.selectbox("Park Proximity", ["No", "Yes"], key="pmr_prox_parc")
            inputs["Prox_Parc"] = 1 if prox_parc_choice == "Yes" else 0
            
            # Metro Proximity as Yes/No (single input, no duplicate)
            prox_metro_choice = st.selectbox("Metro Proximity", ["No", "Yes"], key="pmr_prox_metro")
            inputs["Prox_Metro"] = 1 if prox_metro_choice == "Yes" else 0
        
        elif region_key == "Ste-Rose":
            # Ste-Rose: Remove Garage and Amenagement_paysager from numeric inputs (will be handled as Yes/No)
            num_cols = [c for c in num_cols if c not in ["Garage", "Amenagement_paysager"]]
            
            # Label mapping for Ste-Rose
            label_map = {
                "Etage": "Floor(s)",
                "Age": "Age",
                "Aire_Batiment_m2": "Building Area (m2)",
                "Aire_Lot_m2": "Lot Area (m2)"
            }
            
            # Numeric inputs with renamed labels
            for col in num_cols:
                label = label_map.get(col, col)
                step = 10.0 if any(k in col.lower() for k in ["aire", "m2", "lot"]) else 1.0
                minv = 0.0
                val = st.number_input(label, min_value=float(minv), step=float(step), key=f"ste_rose_{col}")
                inputs[col] = val
            
            # Garage as Yes/No (single input, no duplicate)
            garage_choice = st.selectbox("Garage", ["No", "Yes"], key="ste_rose_garage")
            inputs["Garage"] = 1 if garage_choice == "Yes" else 0
            
            # Landscaping as Yes/No (single input, no duplicate)
            landscaping_choice = st.selectbox("Landscaping", ["No", "Yes"], key="ste_rose_landscaping")
            inputs["Amenagement_paysager"] = 1 if landscaping_choice == "Yes" else 0
        
        else:
            # Fallback for any other regions: use default behavior
            for col in num_cols:
                step = 10.0 if any(k in col.lower() for k in ["aire", "m2", "lot"]) else 1.0
                minv = 0.0 if "age" not in col.lower() else 0.0
                val = st.number_input(col, min_value=float(minv), step=float(step))
                inputs[col] = val

        submitted = st.form_submit_button("Estimate Value")
    
    # Check for missing required fields
    if region_key == "PMR" and "Property_Type" in inputs:
        # Property_Type replaces the old property type columns
        # Also exclude TAXES_AN from missing check (it's not in UI but may be in feature_cols)
        exclude_from_missing = ["CONDO", "5PLEX_ET_MOINS", "6PLEX_ET_PLUS", "UNIFAMILIALE", "TAXES_AN"]
        missing = [c for c in cfg["feature_cols"] if c not in inputs or inputs.get(c) in [None,""]]
        missing = [c for c in missing if c not in exclude_from_missing]
    else:
        missing = [c for c in cfg["feature_cols"] if c not in inputs or inputs.get(c) in [None,""]]
    
    return inputs, submitted, missing

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
        for col in ["Prox_Parc", "Prox_Metro"]:
            if col in df.columns:
                if df[col].dtype == 'object':
                    df[col] = df[col].map({'Yes': 1, 'No': 0, 'True': 1, 'False': 0, True: 1, False: 0}).fillna(0)
    
    # Handle Sainte-Rose comma decimal separators
    if region_key == "Ste-Rose":
        # Identify numeric columns that may contain comma-decimal strings
        numeric_cols = config["num_cols"]
        for col in numeric_cols:
            if col in df.columns:
                if df[col].dtype == "object":
                    # Replace comma with dot and convert to numeric
                    df[col] = df[col].astype(str).str.replace(",", ".", regex=False)
                # Convert to numeric, coercing errors to NaN
                df[col] = pd.to_numeric(df[col], errors="coerce")
    
    # One-hot encode Property_Type if it exists (for PMR or any region)
    if "Property_Type" in df.columns:
        # Define all possible Property_Type values to ensure consistent dummy columns
        possible_types = ["CONDO", "5PLEX_ET_MOINS", "6PLEX_ET_PLUS", "UNIFAMILIALE"]
        # One-hot encode Property_Type into dummy columns
        dummies = pd.get_dummies(df["Property_Type"], prefix="Property_Type")
        # Ensure all possible dummy columns exist (add missing ones with 0)
        for prop_type in possible_types:
            dummy_col = f"Property_Type_{prop_type}"
            if dummy_col not in dummies.columns:
                dummies[dummy_col] = 0
        # Drop the original categorical column and concatenate dummies
        df = pd.concat([df.drop(columns=["Property_Type"]), dummies], axis=1)
        # Remove any original property type indicator columns that might conflict
        original_prop_cols = ["CONDO", "5PLEX_ET_MOINS", "6PLEX_ET_PLUS", "UNIFAMILIALE"]
        for col in original_prop_cols:
            if col in df.columns:
                df = df.drop(columns=[col])
    
    if is_training:
        return df  # keep target
    else:
        df = df.fillna(0)
        # For prediction, return all columns except target (to match training feature_cols)
        if CANON_TARGET in df.columns:
            return df[[c for c in df.columns if c != CANON_TARGET]]
        else:
            return df
    
def train_quantile_models(region_key="BDF"):
    """Train quantile regression models for a specific region using sklearn Pipeline"""
    config = REGION_CONFIG[region_key]
    csv_path = config["data_path"]

    if not csv_path.exists():
        raise FileNotFoundError(f"Dataset introuvable : {csv_path}")
    
    # Special handling for PMR: split into Condo and Plex+SFH submodels
    if region_key == "PMR":
        return train_pmr_segmented_models()
    
    # Standard training for other regions
    with st.spinner(f"Loading and preparing data for {config['name']}..."):
        df = load_region_dataframe_simple(region_key)
        df = create_features(df, region_key, is_training=True)
        if CANON_TARGET not in df.columns:
            raise ValueError(f"[{config['name']}] Target '{CANON_TARGET}' missing after feature prep. Columns: {list(df.columns)}")
        df = df.dropna(subset=[CANON_TARGET])

        # Property_Type is already one-hot encoded by create_features
        # Identify Property_Type dummy columns that were created
        property_type_dummy_cols = [c for c in df.columns if c.startswith("Property_Type_")]

        # Use programmatic feature_cols: all columns except the target
        feature_cols = [c for c in df.columns if c != CANON_TARGET]
        
        # Remove any original property type indicator columns that might conflict
        # (they should have been removed by create_features, but check to be safe)
        original_prop_cols = ["CONDO", "5PLEX_ET_MOINS", "6PLEX_ET_PLUS", "UNIFAMILIALE"]
        feature_cols = [c for c in feature_cols if c not in original_prop_cols]
        
        if not feature_cols:
            raise ValueError(f"[{config['name']}] No usable features. Columns: {list(df.columns)}")

        X = df[feature_cols]
        y = df[CANON_TARGET].astype(float)

    # Build preprocessing pipeline
    # Include Property_Type dummy columns as numeric (they're 0/1)
    num_cols = [col for col in config["num_cols"] if col in feature_cols]
    if property_type_dummy_cols:
        num_cols.extend([col for col in property_type_dummy_cols if col in feature_cols])
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

def train_pmr_segmented_models():
    """Train separate PMR models for Condo and Plex+SFH segments"""
    config = REGION_CONFIG["PMR"]
    csv_path = config["data_path"]
    
    if not csv_path.exists():
        raise FileNotFoundError(f"Dataset introuvable : {csv_path}")
    
    # Load raw PMR data
    with st.spinner("Loading PMR data..."):
        df_raw = load_region_dataframe_simple("PMR")
        
        # Normalize Property_Type values to handle various formats
        if "Property_Type" not in df_raw.columns:
            # Try to infer from old indicator columns
            prop_type_cols = ["CONDO", "5PLEX_ET_MOINS", "6PLEX_ET_PLUS", "UNIFAMILIALE"]
            available_prop_cols = [col for col in prop_type_cols if col in df_raw.columns]
            if available_prop_cols:
                prop_type_df = df_raw[available_prop_cols]
                df_raw["Property_Type"] = prop_type_df.idxmax(axis=1)
                df_raw.loc[prop_type_df.sum(axis=1) == 0, "Property_Type"] = "UNIFAMILIALE"
        
        # Normalize Property_Type values to standard format
        # Map various formats to: "Condo", "Plex", "SFH"
        property_type_mapping = {
            "CONDO": "Condo",
            "Condo": "Condo",
            "condo": "Condo",
            "5PLEX_ET_MOINS": "Plex",
            "6PLEX_ET_PLUS": "Plex",
            "Plex": "Plex",
            "plex": "Plex",
            "UNIFAMILIALE": "SFH",
            "SFH": "SFH",
            "sfh": "SFH",
            "Unifamiliale": "SFH"
        }
        if "Property_Type" in df_raw.columns:
            df_raw["Property_Type"] = df_raw["Property_Type"].map(property_type_mapping).fillna(df_raw["Property_Type"])
        
        # Create two subsets
        df_condo = df_raw[df_raw["Property_Type"] == "Condo"].copy()
        df_plex_sfh = df_raw[df_raw["Property_Type"].isin(["Plex", "SFH"])].copy()
        
        st.info(f"PMR segmentation: {len(df_condo)} Condos, {len(df_plex_sfh)} Plex+SFH")
    
    metrics = {}
    
    # Train Condo models
    if len(df_condo) == 0:
        st.warning("⚠️ No Condo properties found in PMR dataset. Skipping Condo model training.")
        metrics["CONDO_MAE"] = None
    else:
        with st.spinner("Training PMR Condo models (Q05, Q50, Q95)..."):
            try:
                metrics_condo = train_pmr_segment(df_condo, "CONDO")
                metrics["CONDO_MAE"] = metrics_condo.get("MAE", None)
                st.success(f"✅ Condo models saved: PMR_CONDO_Q05.joblib, PMR_CONDO_Q50.joblib, PMR_CONDO_Q95.joblib")
            except Exception as e:
                st.error(f"❌ Failed to train Condo models: {e}")
                metrics["CONDO_MAE"] = None
    
    # Train Plex+SFH models
    if len(df_plex_sfh) == 0:
        st.warning("⚠️ No Plex or SFH properties found in PMR dataset. Skipping Plex+SFH model training.")
        metrics["PLEX_SFH_MAE"] = None
    else:
        with st.spinner("Training PMR Plex+SFH models (Q05, Q50, Q95)..."):
            try:
                metrics_plex_sfh = train_pmr_segment(df_plex_sfh, "PLEX_SFH")
                metrics["PLEX_SFH_MAE"] = metrics_plex_sfh.get("MAE", None)
                st.success(f"✅ Plex+SFH models saved: PMR_PLEX_SFH_Q05.joblib, PMR_PLEX_SFH_Q50.joblib, PMR_PLEX_SFH_Q95.joblib")
            except Exception as e:
                st.error(f"❌ Failed to train Plex+SFH models: {e}")
                metrics["PLEX_SFH_MAE"] = None
    
    # Overall MAE (weighted average)
    if metrics["CONDO_MAE"] is not None and metrics["PLEX_SFH_MAE"] is not None:
        n_condo = len(df_condo)
        n_plex_sfh = len(df_plex_sfh)
        total = n_condo + n_plex_sfh
        metrics["MAE"] = (metrics["CONDO_MAE"] * n_condo + metrics["PLEX_SFH_MAE"] * n_plex_sfh) / total
    
    return metrics

def train_pmr_segment(df_segment, segment_name):
    """Train quantile models for a PMR segment (CONDO or PLEX_SFH)"""
    # Apply preprocessing
    df_proc = create_features(df_segment.copy(), region_key="PMR", is_training=True)
    
    if CANON_TARGET not in df_proc.columns:
        raise ValueError(f"[PMR {segment_name}] Target '{CANON_TARGET}' missing after feature prep. Columns: {list(df_proc.columns)}")
    
    df_proc = df_proc.dropna(subset=[CANON_TARGET])
    
    if len(df_proc) == 0:
        raise ValueError(f"[PMR {segment_name}] No valid samples after preprocessing")
    
    # Get feature columns (all except target)
    feature_cols = [c for c in df_proc.columns if c != CANON_TARGET]
    
    # Remove Property_Type dummy columns (not needed for segmented models)
    feature_cols = [c for c in feature_cols if not c.startswith("Property_Type_")]
    
    # Remove any original property type indicator columns
    original_prop_cols = ["CONDO", "5PLEX_ET_MOINS", "6PLEX_ET_PLUS", "UNIFAMILIALE"]
    feature_cols = [c for c in feature_cols if c not in original_prop_cols]
    
    if not feature_cols:
        raise ValueError(f"[PMR {segment_name}] No usable features. Columns: {list(df_proc.columns)}")
    
    X = df_proc[feature_cols]
    y = df_proc[CANON_TARGET].astype(float)
    
    # Build preprocessing pipeline
    config = REGION_CONFIG["PMR"]
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
    
    segment_metrics = {}
    saved_files = []
    
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
            "region": "PMR",
            "segment": segment_name,
            "num_cols": num_cols,
            "cat_cols": cat_cols
        }
        
        # For the median model (alpha=0.50), also store MAE
        if alpha == 0.50:
            y_pred_val = pipe.predict(X_val)
            segment_metrics["MAE"] = mean_absolute_error(y_val, y_pred_val)
            model_data["mae"] = segment_metrics["MAE"]
        
        # Save pipeline
        path = pmr_submodel_path_for(segment_name, alpha)
        joblib.dump(model_data, path)
        saved_files.append(path.name)
        
        # Verify file was saved
        if not path.exists():
            raise FileNotFoundError(f"Failed to save model file: {path}")
    
    return segment_metrics

def predict_with_models(region_key="BDF", **kwargs):
    """Make predictions using all three quantile models for a specific region"""
    # Load one model to get the feature columns (they should be the same for all quantiles)
    path = model_path_for(region_key, 0.5)
    try:
        data = joblib.load(path)
        feats = data["features"]  # Get feature columns from model metadata
    except FileNotFoundError:
        st.error(f"Model not found for {region_key}. Please train models first.")
        return None, None, None
    
    # Prepare inputs - separate Property_Type from other features
    base_input = {}
    property_type_value = None
    
    for key, value in kwargs.items():
        if key == "Property_Type":
            property_type_value = value
        else:
            base_input[key] = value
    
    # Create base DataFrame with numeric features
    inputs = pd.DataFrame([base_input])
    
    # One-hot encode Property_Type if it exists and model expects Property_Type dummies
    has_property_type_dummies = any(col.startswith("Property_Type_") for col in feats)
    
    if property_type_value is not None and has_property_type_dummies:
        # Create a temporary DataFrame with the property type
        temp_type_df = pd.DataFrame({"Property_Type": [property_type_value]})
        
        # One-hot encode using the same pattern as training
        type_dummies = pd.get_dummies(temp_type_df["Property_Type"], prefix="Property_Type")
        
        # Ensure all possible Property_Type dummy columns exist
        possible_types = ["CONDO", "5PLEX_ET_MOINS", "6PLEX_ET_PLUS", "UNIFAMILIALE"]
        for prop_type in possible_types:
            dummy_col = f"Property_Type_{prop_type}"
            if dummy_col not in type_dummies.columns:
                type_dummies[dummy_col] = 0
        
        # Merge these dummies into the main input row
        inputs = pd.concat([inputs, type_dummies], axis=1)
    
    # Align inputs columns with feature_cols from training
    for col in feats:
        if col not in inputs.columns:
            inputs[col] = 0
    
    # Ensure the column order matches training exactly
    inputs = inputs[feats]

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
    For PMR, only shows 6 user-facing attributes and imputes taxes into base value.
    """
    try:
        # PMR special handling: route to correct submodel
        if region_key == "PMR":
            property_type_ui = kwargs.get("Property_Type")
            if property_type_ui == "CONDO":
                submodel_type = "CONDO"
            else:  # "5PLEX_ET_MOINS" or "UNIFAMILIALE"
                submodel_type = "PLEX_SFH"
            path = pmr_submodel_path_for(submodel_type, 0.5)
        else:
            path = model_path_for(region_key, 0.5)
        
        data = joblib.load(path)
        pipe = data["pipeline"]
        features = data["features"]
        
        # Prepare single-row input as used by the model
        # Separate Property_Type from other features
        base_input = {}
        property_type_value = None
        
        for key, value in kwargs.items():
            if key == "Property_Type":
                property_type_value = value
            else:
                base_input[key] = value
        
        # For PMR submodels, Property_Type is not needed (already segmented)
        if region_key == "PMR":
            # Build input without Property_Type
            inputs = pd.DataFrame([base_input])
        else:
            # Create base DataFrame with numeric features
            inputs = pd.DataFrame([base_input])
            
            # One-hot encode Property_Type if it exists and model expects Property_Type dummies
            has_property_type_dummies = any(col.startswith("Property_Type_") for col in features)
            
            if property_type_value is not None and has_property_type_dummies:
                # Create a temporary DataFrame with the property type
                temp_type_df = pd.DataFrame({"Property_Type": [property_type_value]})
                
                # One-hot encode using the same pattern as training
                type_dummies = pd.get_dummies(temp_type_df["Property_Type"], prefix="Property_Type")
                
                # Ensure all possible Property_Type dummy columns exist
                possible_types = ["CONDO", "5PLEX_ET_MOINS", "6PLEX_ET_PLUS", "UNIFAMILIALE"]
                for prop_type in possible_types:
                    dummy_col = f"Property_Type_{prop_type}"
                    if dummy_col not in type_dummies.columns:
                        type_dummies[dummy_col] = 0
                
                # Merge these dummies into the main input row
                inputs = pd.concat([inputs, type_dummies], axis=1)
        
        # Align inputs columns with feature_cols from training
        for col in features:
            if col not in inputs.columns:
                inputs[col] = 0
        
        # Ensure the column order matches training exactly
        inputs = inputs[features]

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
        
        # PMR special handling: filter to 6 user-facing attributes and impute taxes into base
        if region_key == "PMR":
            # Define the 6 user-facing attributes
            user_facing_attrs = {
                "ETAGES": "Floor(s)",
                "AGE": "Age",
                "AIRE_HABITABLE": "Building Area (m2)",
                "Prox_Parc": "Park Proximity",
                "Prox_Metro": "Metro Proximity"
            }
            # Property Type is already handled by segmentation, so we don't need to show it
            
            # Aggregate SHAP contributions for user-facing attributes
            user_attr_contributions = {}
            taxes_contribution = 0.0
            other_contributions = 0.0
            
            for feature, contribution in zip(feature_names, shap_contributions):
                c = float(contribution)
                if feature in user_facing_attrs:
                    # Use friendly name
                    user_attr_contributions[user_facing_attrs[feature]] = user_attr_contributions.get(user_facing_attrs[feature], 0.0) + c
                elif feature == "TAXES_AN":
                    # Impute taxes into base value
                    taxes_contribution += c
                else:
                    # Other non-user-facing features also go into base
                    other_contributions += c
            
            # Adjust base value to include taxes and other non-user-facing features
            adjusted_base_value = base_value + taxes_contribution + other_contributions
            
            # Create waterfall data with only user-facing attributes
            waterfall_data = []
            cumulative_value = adjusted_base_value
            
            # Base (includes taxes and other non-user features)
            waterfall_data.append({
                'feature': 'Base Value',
                'contribution': 0.0,
                'cumulative': adjusted_base_value,
                'color': '#1f77b4'
            })
            
            # Contributions from user-facing attributes only
            for attr_name, contribution in user_attr_contributions.items():
                c = float(contribution)
                cumulative_value += c
                color = '#2ca02c' if c >= 0 else '#d62728'
                waterfall_data.append({
                    'feature': attr_name,
                    'contribution': c,
                    'cumulative': cumulative_value,
                    'color': color
                })
        else:
            # Standard handling for other regions
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
        
        # Prepare input data - separate Property_Type from other features
        base_input = {}
        property_type_value = None
        
        for key, value in kwargs.items():
            if key == "Property_Type":
                property_type_value = value
            else:
                base_input[key] = value
        
        # Create base DataFrame with numeric features
        inputs = pd.DataFrame([base_input])
        
        # One-hot encode Property_Type if it exists and model expects Property_Type dummies
        has_property_type_dummies = any(col.startswith("Property_Type_") for col in features)
        
        if property_type_value is not None and has_property_type_dummies:
            # Create a temporary DataFrame with the property type
            temp_type_df = pd.DataFrame({"Property_Type": [property_type_value]})
            
            # One-hot encode using the same pattern as training
            type_dummies = pd.get_dummies(temp_type_df["Property_Type"], prefix="Property_Type")
            
            # Ensure all possible Property_Type dummy columns exist
            possible_types = ["CONDO", "5PLEX_ET_MOINS", "6PLEX_ET_PLUS", "UNIFAMILIALE"]
            for prop_type in possible_types:
                dummy_col = f"Property_Type_{prop_type}"
                if dummy_col not in type_dummies.columns:
                    type_dummies[dummy_col] = 0
            
            # Merge these dummies into the main input row
            inputs = pd.concat([inputs, type_dummies], axis=1)
        
        # Align inputs columns with feature_cols from training
        for col in features:
            if col not in inputs.columns:
                inputs[col] = 0
        
        # Ensure the column order matches training exactly
        inputs = inputs[features]
        
        # Transform using pipeline preprocessor
        X_transformed = pipe.named_steps["preproc"].transform(inputs)
        
        # Transform all dataset features
        X_dataset = pipe.named_steps["preproc"].transform(df[features])
        
        # Check for exact match first in original feature space
        exact_match = df.copy()
        # Handle Property_Type separately if it exists
        if "Property_Type" in kwargs and "Property_Type" in df.columns:
            exact_match = exact_match[exact_match["Property_Type"] == kwargs["Property_Type"]]
        # Check other features
        for col in config["feature_cols"]:
            if col in kwargs and col in exact_match.columns:
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
        
        # Build input row - separate Property_Type from other features
        base_input = {}
        property_type_value = None
        
        for key, value in kwargs.items():
            if key == "Property_Type":
                property_type_value = value
            else:
                base_input[key] = value

        # Keep some raw values for explanations
        aire_batiment = float(base_input.get("Aire_Batiment", 0) or 0)
        aire_lot = float(base_input.get("Aire_Lot", 0) or 0)
        age = float(base_input.get("Age", 0) or 0)
        etage = float(base_input.get("Etage", 0) or 0)
        prox_riverain = int(base_input.get("Prox_Riverain", 0) or 0)

        # Create base DataFrame with numeric features
        inputs = pd.DataFrame([base_input])
        
        # One-hot encode Property_Type if it exists and model expects Property_Type dummies
        has_property_type_dummies = any(col.startswith("Property_Type_") for col in features)
        
        if property_type_value is not None and has_property_type_dummies:
            # Create a temporary DataFrame with the property type
            temp_type_df = pd.DataFrame({"Property_Type": [property_type_value]})
            
            # One-hot encode using the same pattern as training
            type_dummies = pd.get_dummies(temp_type_df["Property_Type"], prefix="Property_Type")
            
            # Ensure all possible Property_Type dummy columns exist
            possible_types = ["CONDO", "5PLEX_ET_MOINS", "6PLEX_ET_PLUS", "UNIFAMILIALE"]
            for prop_type in possible_types:
                dummy_col = f"Property_Type_{prop_type}"
                if dummy_col not in type_dummies.columns:
                    type_dummies[dummy_col] = 0
            
            # Merge these dummies into the main input row
            inputs = pd.concat([inputs, type_dummies], axis=1)
        
        # Align inputs columns with feature_cols from training
        for col in features:
            if col not in inputs.columns:
                inputs[col] = 0
        
        # Ensure the column order matches training exactly
        inputs = inputs[features]
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
        if region_key == "PMR":
            model_exists = models_available(region_key)
        else:
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
            # Show correct feature count based on region
            if region_key == "Ste-Rose":
                feature_count = "6 Raw Attributes"
            else:
                feature_count = "5 Raw Attributes"
            st.metric("Features", feature_count, "Raw attributes for more relevant insights")
    
    elif page == "Property Valuation":
        if not models_available(region_key):
            st.info("Models for this region are not available yet. Train locally using `python train_models.py`, then commit and push the generated files in `/models` on your working branch. After deployment, come back here to estimate.")
            st.stop()

        inputs, submitted, missing = build_input_form(region_key)
        if submitted:
            if missing:
                st.error(f"Please provide all required inputs: {missing}")
                st.stop()
            # Load q05/q50/q95 and predict
            try:
                # PMR routing: determine which submodel to use
                if region_key == "PMR":
                    # Get Property_Type from inputs (UI value: "Condo", "Plex", "SFH")
                    property_type_ui = inputs.get("Property_Type")
                    if property_type_ui is None:
                        st.error("Property Type is required for PMR predictions")
                        st.stop()
                    
                    # Map UI value to internal value for routing
                    # The UI sends "CONDO", "5PLEX_ET_MOINS", or "UNIFAMILIALE" based on build_input_form
                    # But we need to map to segment names
                    if property_type_ui == "CONDO":
                        submodel_type = "CONDO"
                    else:  # "5PLEX_ET_MOINS" or "UNIFAMILIALE" -> Plex or SFH
                        submodel_type = "PLEX_SFH"
                    
                    # Load the appropriate PMR submodel
                    q05_path = pmr_submodel_path_for(submodel_type, 0.05)
                    q50_path = pmr_submodel_path_for(submodel_type, 0.5)
                    q95_path = pmr_submodel_path_for(submodel_type, 0.95)
                    
                    data50 = joblib.load(q50_path)
                    data05 = joblib.load(q05_path)
                    data95 = joblib.load(q95_path)
                    pipe50 = data50["pipeline"]
                    feats = data50["features"]
                    
                    # For PMR submodels, Property_Type is not needed (already segmented)
                    # Build base input dict without Property_Type
                    base_input = {}
                    for key, value in inputs.items():
                        if key != "Property_Type":
                            base_input[key] = value
                    
                    # Create base DataFrame with numeric features
                    X_input = pd.DataFrame([base_input])
                    
                else:
                    # Standard routing for other regions
                    data50 = joblib.load(model_path_for(region_key, 0.5))
                    data05 = joblib.load(model_path_for(region_key, 0.05))
                    data95 = joblib.load(model_path_for(region_key, 0.95))
                    pipe50 = data50["pipeline"]
                    feats = data50["features"]
                    
                    # Build base input dict with numeric features (excluding Property_Type for now)
                    base_input = {}
                    property_type_value = None
                    
                    for key, value in inputs.items():
                        if key == "Property_Type":
                            property_type_value = value
                        else:
                            base_input[key] = value
                    
                    # Create base DataFrame with numeric features
                    X_input = pd.DataFrame([base_input])
                    
                    # One-hot encode Property_Type if it exists (for other regions with Property_Type)
                    # Check if Property_Type dummy columns are expected in the model features
                    has_property_type_dummies = any(col.startswith("Property_Type_") for col in feats)
                    
                    if property_type_value is not None and has_property_type_dummies:
                        # Create a temporary DataFrame with the property type
                        temp_type_df = pd.DataFrame({"Property_Type": [property_type_value]})
                        
                        # One-hot encode using the same pattern as training
                        type_dummies = pd.get_dummies(temp_type_df["Property_Type"], prefix="Property_Type")
                        
                        # Ensure all possible Property_Type dummy columns exist (add missing ones with 0)
                        # Use the same possible types as in training
                        possible_types = ["CONDO", "5PLEX_ET_MOINS", "6PLEX_ET_PLUS", "UNIFAMILIALE"]
                        for prop_type in possible_types:
                            dummy_col = f"Property_Type_{prop_type}"
                            if dummy_col not in type_dummies.columns:
                                type_dummies[dummy_col] = 0
                        
                        # Merge these dummies into the main input row
                        X_input = pd.concat([X_input, type_dummies], axis=1)
                
                # Align X_input columns with feature_cols from training
                # Add any missing columns (e.g., some dummy types not present for that selection) with 0
                for col in feats:
                    if col not in X_input.columns:
                        X_input[col] = 0
                
                # Ensure the column order matches training exactly
                X_input = X_input[feats]
                
                y50 = float(pipe50.predict(X_input)[0])
                y05 = float(data05["pipeline"].predict(X_input)[0])
                y95 = float(data95["pipeline"].predict(X_input)[0])

                st.success(f"Estimated property value (median): ${y50:,.0f}")
                st.write(f"Range (q05–q95): ${y05:,.0f} – ${y95:,.0f}")

                # MAE if present
                mae = data50.get("mae")
                if mae is not None:
                    segment_info = f" ({data50.get('segment', '')})" if region_key == "PMR" else ""
                    st.caption(f"Model MAE (validation): ${mae:,.0f}{segment_info}")

                # SHAP / Waterfall (keep your existing functions; wrap in try/except)
                try:
                    fig = create_shap_waterfall_chart(region_key=region_key, predicted_value=y50, **inputs)
                    if fig:
                        st.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    st.warning(f"SHAP visualization unavailable: {e}")
                    
            except Exception as e:
                st.error(f"Prediction failed: {e}")
    

    
    elif page == "Model Performance":
        st.header(f"📈 Model Training & Performance - {selected_region['name']}")
        
        # Check if models exist for this region
        if region_key == "PMR":
            model_path = pmr_submodel_path_for("CONDO", 0.5)  # Check one submodel as indicator
        else:
            model_path = model_path_for(region_key, 0.5)
        
        if st.button(f"🚀 Train Quantile Models for {selected_region['name']}", type="primary"):
            try:
                with st.spinner(f"Training models for {selected_region['name']}..."):
                    metrics = train_quantile_models(region_key)
                
                st.success(f"✅ Models for {selected_region['name']} trained and saved successfully!")
                
                # Display metrics
                if region_key == "PMR":
                    # Show segment-specific MAEs
                    if "CONDO_MAE" in metrics and metrics["CONDO_MAE"] is not None:
                        st.metric("Condo MAE", f"${metrics['CONDO_MAE']:,.0f}")
                    if "PLEX_SFH_MAE" in metrics and metrics["PLEX_SFH_MAE"] is not None:
                        st.metric("Plex+SFH MAE", f"${metrics['PLEX_SFH_MAE']:,.0f}")
                    if "MAE" in metrics and metrics["MAE"] is not None:
                        st.metric("Overall MAE (weighted)", f"${metrics['MAE']:,.0f}")
                        st.markdown("*Weighted average across segments*")
                else:
                    if "MAE" in metrics:
                        st.metric("Mean Absolute Error", f"${metrics['MAE']:,.0f}")
                        st.markdown("*Average prediction error*")
                
                
                # Temporary Model Downloads
                st.subheader("Temporary Model Downloads (this branch only)")
                alpha_map = {0.05: "q05", 0.50: "q50", 0.95: "q95"}
                
                if region_key == "PMR":
                    # Show downloads for both PMR submodels
                    for submodel in ["CONDO", "PLEX_SFH"]:
                        st.write(f"**{submodel} Models:**")
                        for alpha in [0.05, 0.50, 0.95]:
                            path = pmr_submodel_path_for(submodel, alpha)
                            if path.exists():
                                st.download_button(
                                    label=f"⬇️ Download {submodel} model ({alpha_map[alpha]})",
                                    data=_read_binary(path),
                                    file_name=path.name,
                                    mime="application/octet-stream",
                                    key=f"pmr_{submodel}_{alpha}"
                                )
                else:
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
                
                # For PMR, compute Overall MAE from full dataset
                if region_key == "PMR" and "mae" in data:
                    # This is segment-specific MAE, we'll compute Overall MAE separately below
                    pass
                elif "mae" in data:
                    st.write(f"- MAE: ${data['mae']:,.0f}")
                    
            except Exception as e:
                st.warning(f"Model files exist but may be corrupted: {e}. Please retrain.")
            
            # For PMR, show submodel info and compute Overall MAE
            if region_key == "PMR" and models_available("PMR"):
                try:
                    # Compute Overall MAE from full dataset
                    pmr_config = REGION_CONFIG["PMR"]
                    pmr_csv_path = pmr_config["data_path"]
                    
                    if pmr_csv_path.exists():
                        # Load raw PMR data
                        df_raw_pmr = load_region_dataframe_simple("PMR")
                        
                        # Normalize Property_Type
                        if "Property_Type" not in df_raw_pmr.columns:
                            prop_type_cols = ["CONDO", "5PLEX_ET_MOINS", "6PLEX_ET_PLUS", "UNIFAMILIALE"]
                            available_prop_cols = [col for col in prop_type_cols if col in df_raw_pmr.columns]
                            if available_prop_cols:
                                prop_type_df = df_raw_pmr[available_prop_cols]
                                df_raw_pmr["Property_Type"] = prop_type_df.idxmax(axis=1)
                                df_raw_pmr.loc[prop_type_df.sum(axis=1) == 0, "Property_Type"] = "UNIFAMILIALE"
                        
                        property_type_mapping = {
                            "CONDO": "Condo", "Condo": "Condo", "condo": "Condo",
                            "5PLEX_ET_MOINS": "Plex", "6PLEX_ET_PLUS": "Plex", "Plex": "Plex",
                            "UNIFAMILIALE": "SFH", "SFH": "SFH", "sfh": "SFH"
                        }
                        if "Property_Type" in df_raw_pmr.columns:
                            df_raw_pmr["Property_Type"] = df_raw_pmr["Property_Type"].map(property_type_mapping).fillna(df_raw_pmr["Property_Type"])
                        
                        # Apply preprocessing
                        df_proc_pmr = create_features(df_raw_pmr.copy(), region_key="PMR", is_training=True)
                        df_proc_pmr = df_proc_pmr.dropna(subset=[CANON_TARGET])
                        
                        y_true_pmr = df_proc_pmr[CANON_TARGET].astype(float)
                        y_pred_pmr_list = []
                        
                        # Route to correct submodel and predict
                        if "Property_Type" in df_raw_pmr.columns:
                            property_type_orig = df_raw_pmr["Property_Type"].loc[df_proc_pmr.index]
                            
                            for segment_type in ["Condo", "Plex", "SFH"]:
                                segment_mask = property_type_orig == segment_type
                                if segment_mask.sum() == 0:
                                    continue
                                
                                if segment_type == "Condo":
                                    submodel_type = "CONDO"
                                else:
                                    submodel_type = "PLEX_SFH"
                                
                                pmr_model_path = pmr_submodel_path_for(submodel_type, 0.5)
                                model_data_seg = joblib.load(pmr_model_path)
                                pipe_seg = model_data_seg["pipeline"]
                                feature_cols_seg = model_data_seg["features"]
                                
                                df_segment = df_proc_pmr[segment_mask].copy()
                                
                                for col in feature_cols_seg:
                                    if col not in df_segment.columns:
                                        df_segment[col] = 0
                                
                                X_segment = df_segment[feature_cols_seg]
                                y_pred_segment = pipe_seg.predict(X_segment)
                                
                                segment_indices = df_segment.index
                                for idx, pred in zip(segment_indices, y_pred_segment):
                                    y_pred_pmr_list.append((idx, pred))
                            
                            y_pred_pmr_list.sort(key=lambda x: x[0])
                            y_pred_pmr = np.array([pred for _, pred in y_pred_pmr_list])
                        else:
                            # Fallback
                            pmr_model_path = pmr_submodel_path_for("CONDO", 0.5)
                            model_data_seg = joblib.load(pmr_model_path)
                            pipe_seg = model_data_seg["pipeline"]
                            feature_cols_seg = model_data_seg["features"]
                            
                            for col in feature_cols_seg:
                                if col not in df_proc_pmr.columns:
                                    df_proc_pmr[col] = 0
                            
                            X_pmr = df_proc_pmr[feature_cols_seg]
                            y_pred_pmr = pipe_seg.predict(X_pmr)
                        
                        # Compute Overall MAE
                        overall_mae_pmr = mean_absolute_error(y_true_pmr, y_pred_pmr)
                        
                        # Display Overall MAE in Model Details
                        st.write(f"- Overall MAE: ${overall_mae_pmr:,.0f}")
                    
                    # Show submodel details
                    for submodel in ["CONDO", "PLEX_SFH"]:
                        submodel_path = pmr_submodel_path_for(submodel, 0.5)
                        if submodel_path.exists():
                            data = joblib.load(submodel_path)
                            st.info(f"**{submodel} Model Details:**")
                            st.write(f"- Features: {len(data['features'])} attributes")
                            st.write(f"- Segment: {data.get('segment', 'Unknown')}")
                            if "mae" in data:
                                st.write(f"- Segment MAE: ${data['mae']:,.0f}")
                except Exception as e:
                    st.warning(f"Could not load PMR submodel info or compute Overall MAE: {e}")
            
            # PMR-specific evaluation: Overall MAE and MAE by Property Type
            if region_key == "PMR" and models_available("PMR"):
                try:
                    st.divider()
                    st.subheader("PMR – Overall MAE and MAE by Property Type")
                    
                    # Load raw PMR data
                    pmr_config = REGION_CONFIG["PMR"]
                    pmr_csv_path = pmr_config["data_path"]
                    
                    if not pmr_csv_path.exists():
                        st.warning(f"PMR dataset not found at {pmr_csv_path}")
                    else:
                        # Load raw data
                        df_raw = load_region_dataframe_simple("PMR")
                        
                        # Preserve original Property_Type for grouping (before preprocessing)
                        if "Property_Type" not in df_raw.columns:
                            # If Property_Type doesn't exist, try to infer from old indicator columns
                            prop_type_cols = ["CONDO", "5PLEX_ET_MOINS", "6PLEX_ET_PLUS", "UNIFAMILIALE"]
                            available_prop_cols = [col for col in prop_type_cols if col in df_raw.columns]
                            
                            if available_prop_cols:
                                # Create Property_Type from old indicator columns
                                # Use idxmax to find which column has value 1 (they should be mutually exclusive)
                                prop_type_df = df_raw[available_prop_cols]
                                df_raw["Property_Type"] = prop_type_df.idxmax(axis=1)
                                # Handle cases where all columns are 0 (default to UNIFAMILIALE)
                                df_raw.loc[prop_type_df.sum(axis=1) == 0, "Property_Type"] = "UNIFAMILIALE"
                        
                        # Store Property_Type for later grouping
                        property_type_original = df_raw["Property_Type"].copy() if "Property_Type" in df_raw.columns else None
                        
                        # Identify target column
                        target_col = CANON_TARGET
                        if target_col not in df_raw.columns:
                            # Try alternative names
                            for alt in ["Prix_de_Vente", "prix_de_vente", "Prix", "Price", "price_sold"]:
                                if alt in df_raw.columns:
                                    df_raw = df_raw.rename(columns={alt: target_col})
                                    break
                        
                        if target_col not in df_raw.columns:
                            st.error(f"Target column '{target_col}' not found in PMR dataset. Available columns: {list(df_raw.columns)}")
                        else:
                            # Apply same preprocessing as in training
                            df_proc = create_features(df_raw.copy(), region_key="PMR", is_training=True)
                            
                            # Drop rows with missing target
                            df_proc = df_proc.dropna(subset=[target_col])
                            if property_type_original is not None:
                                property_type_original = property_type_original.loc[df_proc.index]
                            
                            # For PMR, use segmented models to compute Overall MAE
                            # We need to route each property to the correct submodel
                            y_true = df_proc[target_col].astype(float)
                            y_pred_list = []
                            
                            # Normalize Property_Type for routing
                            property_type_mapping = {
                                "CONDO": "Condo",
                                "Condo": "Condo",
                                "condo": "Condo",
                                "5PLEX_ET_MOINS": "Plex",
                                "6PLEX_ET_PLUS": "Plex",
                                "Plex": "Plex",
                                "UNIFAMILIALE": "SFH",
                                "SFH": "SFH"
                            }
                            
                            if property_type_original is not None:
                                df_proc["Property_Type_Normalized"] = property_type_original.map(property_type_mapping).fillna(property_type_original)
                                
                                # Split by segment and predict with appropriate model
                                for segment_type in ["Condo", "Plex", "SFH"]:
                                    segment_mask = df_proc["Property_Type_Normalized"] == segment_type
                                    if segment_mask.sum() == 0:
                                        continue
                                    
                                    # Determine submodel
                                    if segment_type == "Condo":
                                        submodel_type = "CONDO"
                                    else:
                                        submodel_type = "PLEX_SFH"
                                    
                                    # Load appropriate model
                                    pmr_model_path = pmr_submodel_path_for(submodel_type, 0.5)
                                    model_data = joblib.load(pmr_model_path)
                                    pipe = model_data["pipeline"]
                                    feature_cols = model_data["features"]
                                    
                                    # Get segment data
                                    df_segment = df_proc[segment_mask].copy()
                                    
                                    # Ensure df_segment has all feature columns expected by the model
                                    for col in feature_cols:
                                        if col not in df_segment.columns:
                                            df_segment[col] = 0
                                    
                                    # Select features in the correct order
                                    X_segment = df_segment[feature_cols]
                                    
                                    # Make predictions
                                    y_pred_segment = pipe.predict(X_segment)
                                    
                                    # Store predictions in correct order
                                    segment_indices = df_segment.index
                                    for idx, pred in zip(segment_indices, y_pred_segment):
                                        y_pred_list.append((idx, pred))
                                
                                # Sort by original index and extract predictions
                                y_pred_list.sort(key=lambda x: x[0])
                                y_pred = np.array([pred for _, pred in y_pred_list])
                            else:
                                # Fallback: use first available submodel
                                pmr_model_path = pmr_submodel_path_for("CONDO", 0.5)
                                model_data = joblib.load(pmr_model_path)
                                pipe = model_data["pipeline"]
                                feature_cols = model_data["features"]
                                
                                # Ensure df_proc has all feature columns expected by the model
                                for col in feature_cols:
                                    if col not in df_proc.columns:
                                        df_proc[col] = 0
                                
                                # Select features in the correct order
                                X = df_proc[feature_cols]
                                
                                # Make predictions using the pipeline
                                y_pred = pipe.predict(X)
                            
                            # Compute MAE overall and by Property_Type
                            df_eval = pd.DataFrame({
                                "y_true": y_true.values,
                                "y_pred": y_pred,
                                "abs_err": np.abs(y_true.values - y_pred)
                            })
                            
                            # Add Property_Type for grouping if available
                            if property_type_original is not None:
                                df_eval["Property_Type"] = property_type_original.values
                                
                                # Map Property_Type values to UI-friendly names
                                property_type_map = {
                                    "CONDO": "Condo",
                                    "5PLEX_ET_MOINS": "Plex",
                                    "6PLEX_ET_PLUS": "Plex",  # Map both plex types to "Plex"
                                    "UNIFAMILIALE": "SFH"
                                }
                                df_eval["Property_Type_UI"] = df_eval["Property_Type"].map(property_type_map).fillna(df_eval["Property_Type"])
                                
                                # Compute overall MAE
                                overall_mae = df_eval["abs_err"].mean()
                                
                                # Compute MAE by Property Type (using UI-friendly names)
                                mae_by_type = (
                                    df_eval
                                    .groupby("Property_Type_UI")["abs_err"]
                                    .mean()
                                    .reset_index()
                                    .rename(columns={"Property_Type_UI": "Property Type", "abs_err": "MAE"})
                                )
                                mae_by_type["MAE"] = mae_by_type["MAE"].round(0).astype(int)
                                
                                # Display results
                                col1, col2 = st.columns(2)
                                with col1:
                                    st.metric("Overall MAE", f"${overall_mae:,.0f}")
                                with col2:
                                    st.metric("Number of Properties", len(df_eval))
                                
                                st.subheader("MAE by Property Type")
                                st.dataframe(mae_by_type, use_container_width=True)
                                
                                # Show sample counts by property type
                                st.caption(f"Sample sizes: {df_eval.groupby('Property_Type_UI').size().to_dict()}")
                            else:
                                # If Property_Type is not available, just show overall MAE
                                overall_mae = df_eval["abs_err"].mean()
                                st.metric("Overall MAE", f"${overall_mae:,.0f}")
                                st.info("Property Type information not available in the dataset for segment analysis.")
                                
                except Exception as e:
                    st.warning(f"Could not compute PMR evaluation metrics: {e}")
        
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
