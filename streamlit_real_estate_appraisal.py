# Streamlit Property Valuation App with Quantile Regression
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_absolute_error
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import shap
import warnings
warnings.filterwarnings('ignore')

# ---------------- CONFIG ----------------
# Base directory of the current script
BASE_DIR = Path(__file__).parent

# Paths to dataset and quantile models
DATA_PATH = BASE_DIR / "donnees_BDF.csv"
MODEL_Q05_PATH = BASE_DIR / "property_model_q05.joblib"
MODEL_Q50_PATH = BASE_DIR / "property_model_q50.joblib"
MODEL_Q95_PATH = BASE_DIR / "property_model_q95.joblib"

# Load dataset
try:
    df = pd.read_csv(DATA_PATH)
except FileNotFoundError:
    st.error(f"Dataset introuvable : {DATA_PATH}")

# Load quantile models
try:
    model_q05 = joblib.load(MODEL_Q05_PATH)
    model_q50 = joblib.load(MODEL_Q50_PATH)
    model_q95 = joblib.load(MODEL_Q95_PATH)
except FileNotFoundError as e:
    st.error(f"Model introuvable : {e.filename}")


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
</style>
""", unsafe_allow_html=True)

# ---------------- FONCTIONS ----------------
def create_features(df, is_training=True):
    """Prepare raw data features"""
    df = df.copy()
    
    # Ensure all required columns exist
    required_cols = ["Etage", "Age", "Aire_Batiment", "Aire_Lot", "Prox_Riverain"]
    for col in required_cols:
        if col not in df.columns:
            df[col] = 0
    
    if not is_training:
        df = df.fillna(0)
    
    return df

def train_quantile_models(csv_path):
    """Train quantile regression models"""
    if not csv_path.exists():
        raise FileNotFoundError(f"Dataset introuvable : {csv_path}")
    
    with st.spinner("Loading and preparing data..."):
        df = pd.read_csv(csv_path)
        df = create_features(df, is_training=True)
        df = df.dropna(subset=['Prix_de_vente'])

    feature_cols = ["Etage", "Age", "Aire_Batiment", "Aire_Lot", "Prox_Riverain"]
    X = df[feature_cols].fillna(0)
    y = df['Prix_de_vente']

    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X)

    metrics = {}
    models = {}
    
    with st.spinner("Training quantile models..."):
        for alpha, path in zip([0.05, 0.50, 0.95],
                               [MODEL_Q05_PATH, MODEL_Q50_PATH, MODEL_Q95_PATH]):
            X_train, X_val, y_train, y_val = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
            gbr = GradientBoostingRegressor(loss="quantile", alpha=alpha, random_state=42)
            gbr.fit(X_train, y_train)
            
            # Store model data
            model_data = {"model": gbr, "scaler": scaler, "features": feature_cols}
            
            # For the median model (alpha=0.50), also store MAE
            if alpha == 0.50:
                y_pred_val = gbr.predict(X_val)
                metrics["R2"] = r2_score(y_val, y_pred_val)
                metrics["MAE"] = mean_absolute_error(y_val, y_pred_val)
                model_data["mae"] = metrics["MAE"]
            
            joblib.dump(model_data, path)
            models[alpha] = gbr

    return metrics

def predict_with_models(etage, age, aire_batiment, aire_lot, prox_riverain):
    """Make predictions using all three quantile models"""
    inputs = pd.DataFrame([{
        "Etage": etage,
        "Age": age,
        "Aire_Batiment": aire_batiment,
        "Aire_Lot": aire_lot,
        "Prox_Riverain": prox_riverain
    }])
    inputs = create_features(inputs, is_training=False)

    preds = {}
    for alpha, path in zip([0.05, 0.50, 0.95],
                           [MODEL_Q05_PATH, MODEL_Q50_PATH, MODEL_Q95_PATH]):
        data = joblib.load(path)
        scaler = data["scaler"]
        features = data["features"]
        model = data["model"]
        X_scaled = scaler.transform(inputs[features])
        preds[alpha] = float(model.predict(X_scaled)[0])

    return preds[0.05], preds[0.50], preds[0.95]

def get_model_mae():
    """Get the MAE from the trained model"""
    try:
        # Load the median model to get the MAE
        data = joblib.load(MODEL_Q50_PATH)
        if "mae" in data:
            return data["mae"]
        else:
            # If MAE is not stored in the model, compute it from the training data
            df = pd.read_csv(DATA_PATH)
            df = create_features(df, is_training=True)
            df = df.dropna(subset=['Prix_de_vente'])
            
            feature_cols = ["Etage", "Age", "Aire_Batiment", "Aire_Lot", "Prox_Riverain"]
            X = df[feature_cols].fillna(0)
            y = df['Prix_de_vente']
            
            scaler = data["scaler"]
            model = data["model"]
            X_scaled = scaler.transform(X)
            y_pred = model.predict(X_scaled)
            
            return mean_absolute_error(y, y_pred)
    except Exception as e:
        st.warning(f"Could not retrieve MAE: {e}")
        return None

def create_shap_waterfall_chart(etage, age, aire_batiment, aire_lot, prox_riverain, predicted_value):
    """
    Create a SHAP waterfall chart showing how the predicted price is constructed.
    
    This function computes SHAP values for the current property and creates a waterfall chart
    that shows how each feature contributes to the final prediction, starting from the base value
    (average predicted price in the training dataset) and adding/subtracting contributions.
    
    Args:
        etage, age, aire_batiment, aire_lot, prox_riverain: Property features
        predicted_value: The final predicted value from the model
        
    Returns:
        plotly.graph_objects.Figure: Waterfall chart showing price construction
    """
    try:
        # Load the median model for SHAP analysis
        data = joblib.load(MODEL_Q50_PATH)
        model = data["model"]
        scaler = data["scaler"]
        features = data["features"]
        
        # Prepare input data
        inputs = pd.DataFrame([{
            "Etage": etage,
            "Age": age,
            "Aire_Batiment": aire_batiment,
            "Aire_Lot": aire_lot,
            "Prox_Riverain": prox_riverain
        }])
        inputs = create_features(inputs, is_training=False)
        X_scaled = scaler.transform(inputs[features])
        
        # Compute SHAP values using TreeExplainer for GradientBoostingRegressor
        # SHAP values represent the contribution of each feature to the prediction
        # Positive values increase the predicted price, negative values decrease it
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_scaled)
        
        # Get the base value (expected value) from the explainer
        # This represents the average prediction across the training dataset
        # Convert numpy scalar to Python float to avoid formatting issues
        base_value = float(explainer.expected_value)
        
        # Prepare data for waterfall chart
        feature_names = features
        shap_contributions = shap_values[0]  # First (and only) prediction
        
        # Create waterfall data
        waterfall_data = []
        cumulative_value = base_value
        
        # Add base value
        waterfall_data.append({
            'feature': 'Base Value',
            'contribution': 0,
            'cumulative': base_value,
            'color': '#1f77b4'  # Blue for base value
        })
        
        # Add each feature contribution
        for i, (feature, contribution) in enumerate(zip(feature_names, shap_contributions)):
            # Convert numpy contribution to Python float to avoid formatting issues
            contribution_float = float(contribution)
            cumulative_value += contribution_float
            
            # Color coding: green for positive contribution, red for negative
            color = '#2ca02c' if contribution_float >= 0 else '#d62728'
            
            waterfall_data.append({
                'feature': feature,
                'contribution': contribution_float,
                'cumulative': cumulative_value,
                'color': color
            })
        
        # Add final value
        waterfall_data.append({
            'feature': 'Final Prediction',
            'contribution': 0,
            'cumulative': predicted_value,
            'color': '#1f77b4'  # Blue for final value
        })
        
        # Create the waterfall chart using Plotly
        fig = go.Figure()
        
        # Add bars for each step
        for i, data_point in enumerate(waterfall_data):
            if i == 0 or i == len(waterfall_data) - 1:
                # Base value and final prediction - show as full bars
                # Convert to float to ensure proper string formatting
                cumulative_val = float(data_point['cumulative'])
                fig.add_trace(go.Bar(
                    x=[data_point['feature']],
                    y=[cumulative_val],
                    marker_color=data_point['color'],
                    name=data_point['feature'],
                    text=[f"${cumulative_val:,.0f}"],
                    textposition='auto',
                    showlegend=False
                ))
            else:
                # Feature contributions - show as incremental bars
                # Convert to float to ensure proper string formatting
                contribution_val = float(data_point['contribution'])
                fig.add_trace(go.Bar(
                    x=[data_point['feature']],
                    y=[contribution_val],
                    marker_color=data_point['color'],
                    name=data_point['feature'],
                    text=[f"${contribution_val:,.0f}"],
                    textposition='auto',
                    showlegend=False
                ))
        
        # Add connecting lines to show the flow
        x_positions = list(range(len(waterfall_data)))
        cumulative_values = [float(point['cumulative']) for point in waterfall_data]
        
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
        
        # Update layout
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
                ticktext=[point['feature'] for point in waterfall_data]
            ),
            yaxis=dict(
                tickformat='$,.0f'
            ),
            margin=dict(b=100)  # Add bottom margin for rotated labels
        )
        
        return fig
        
    except Exception as e:
        st.warning(f"Could not create SHAP waterfall chart: {e}")
        return None

def find_nearest_neighbors_and_calculate_ratio(etage, age, aire_batiment, aire_lot, prox_riverain, predicted_value, k=3):
    """
    Find nearest neighbors in the dataset and calculate the ratio between predicted and actual prices.
    
    Args:
        etage, age, aire_batiment, aire_lot, prox_riverain: Property features
        predicted_value: The predicted value from the model
        k: Number of nearest neighbors to consider (default=3)
        
    Returns:
        dict: Contains ratio, actual_price, and whether averaging was used
    """
    try:
        # Load the dataset
        df = pd.read_csv(DATA_PATH)
        df = create_features(df, is_training=True)
        df = df.dropna(subset=['Prix_de_vente'])
        
        # Prepare input features for similarity search
        # Note: We exclude 'Prix_de_vente' from the similarity search because we want to find
        # properties with similar characteristics (explanatory variables) but use their actual
        # sale prices as ground truth for comparison. This prevents data leakage where the
        # target variable would influence the neighbor selection.
        feature_cols = ["Etage", "Age", "Aire_Batiment", "Aire_Lot", "Prox_Riverain"]
        
        # Prepare input vector
        input_vector = np.array([etage, age, aire_batiment, aire_lot, prox_riverain])
        
        # Check for exact match first
        exact_match = df[
            (df['Etage'] == etage) & 
            (df['Age'] == age) & 
            (df['Aire_Batiment'] == aire_batiment) & 
            (df['Aire_Lot'] == aire_lot) & 
            (df['Prox_Riverain'] == prox_riverain)
        ]
        
        if len(exact_match) > 0:
            # Use the first exact match
            actual_price = exact_match['Prix_de_vente'].iloc[0]
            used_averaging = False
        else:
            # Find k nearest neighbors using Euclidean distance
            # We use only the explanatory variables for distance calculation
            X_features = df[feature_cols].values
            
            # Calculate distances
            distances = np.sqrt(np.sum((X_features - input_vector) ** 2, axis=1))
            
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
        st.warning(f"Could not calculate ratio: {e}")
        return None

def compute_shap_values(etage, age, aire_batiment, aire_lot, prox_riverain):
    """
    Compute SHAP values for property prediction using the trained GradientBoostingRegressor model.
    
    SHAP (SHapley Additive exPlanations) values explain how each feature contributes to the prediction.
    Each SHAP value represents the contribution of a feature to the final prediction in dollars.
    
    Args:
        etage, age, aire_batiment, aire_lot, prox_riverain: Property features
        
    Returns:
        dict: SHAP values for each feature with explanations
    """
    try:
        # Load the median model (50th percentile) for SHAP analysis
        data = joblib.load(MODEL_Q50_PATH)
        model = data["model"]
        scaler = data["scaler"]
        features = data["features"]
        
        # Prepare input data
        inputs = pd.DataFrame([{
            "Etage": etage,
            "Age": age,
            "Aire_Batiment": aire_batiment,
            "Aire_Lot": aire_lot,
            "Prox_Riverain": prox_riverain
        }])
        inputs = create_features(inputs, is_training=False)
        X_scaled = scaler.transform(inputs[features])
        
        # Create SHAP explainer for GradientBoostingRegressor
        # Using TreeExplainer which is optimized for tree-based models like GradientBoostingRegressor
        # TreeExplainer provides exact SHAP values for tree models, making it faster and more accurate
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_scaled)
        
        # Get feature names
        feature_names = features
        
        # Create SHAP analysis results
        shap_results = {}
        
        for i, feature in enumerate(feature_names):
            shap_value = shap_values[0][i]  # First (and only) prediction
            
            # Create explanations based on feature type
            if feature == "Aire_Batiment":
                # Building area: show contribution per m²
                contribution_per_m2 = shap_value / aire_batiment if aire_batiment > 0 else 0
                shap_results[feature] = {
                    "shap_value": shap_value,
                    "description": f"Each extra 10 m² adds ~${contribution_per_m2 * 10:,.0f}",
                    "label": "Building Efficiency"
                }
                
            elif feature == "Age":
                # Age: modern vs old comparison
                modern_threshold = 20
                if age < modern_threshold:
                    shap_results[feature] = {
                        "shap_value": shap_value,
                        "description": f"Modern building (Age < {modern_threshold}) adds ${shap_value:,.0f}",
                        "label": "Condition"
                    }
                else:
                    shap_results[feature] = {
                        "shap_value": shap_value,
                        "description": f"Older building (Age ≥ {modern_threshold}) reduces value by ${abs(shap_value):,.0f}",
                        "label": "Condition"
                    }
                    
            elif feature == "Prox_Riverain":
                # Waterfront proximity
                if prox_riverain == 1:
                    shap_results[feature] = {
                        "shap_value": shap_value,
                        "description": f"Waterfront location adds ${shap_value:,.0f}",
                        "label": "Premium Location"
                    }
                else:
                    shap_results[feature] = {
                        "shap_value": shap_value,
                        "description": f"Non-waterfront location: ${shap_value:,.0f} impact",
                        "label": "Premium Location"
                    }
                    
            elif feature == "Etage":
                # Floor level
                if etage > 1:
                    shap_results[feature] = {
                        "shap_value": shap_value,
                        "description": f"Multi-floor ({etage} floors) adds ${shap_value:,.0f}",
                        "label": "Floor Level"
                    }
                else:
                    shap_results[feature] = {
                        "shap_value": shap_value,
                        "description": f"Single floor: ${shap_value:,.0f} impact",
                        "label": "Floor Level"
                    }
                    
            elif feature == "Aire_Lot":
                # Lot area: show contribution per m²
                contribution_per_m2 = shap_value / aire_lot if aire_lot > 0 else 0
                shap_results[feature] = {
                    "shap_value": shap_value,
                    "description": f"Each extra 10 m² adds ~${contribution_per_m2 * 10:,.0f}",
                    "label": "Lot Size"
                }
        
        return shap_results
        
    except Exception as e:
        st.warning(f"Could not compute SHAP values: {e}")
        return None



def main():
    # Header
    st.markdown('<h1 class="main-header">🏠 Property Valuation System</h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; font-size: 1.2rem; color: #666;">Gradient Boosting with Quantile Loss for Accurate Property Valuations</p>', unsafe_allow_html=True)
    
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
        if MODEL_Q50_PATH.exists():
            st.markdown('<div class="success-box">', unsafe_allow_html=True)
            st.markdown("✅ **Model Status**: Quantile models are trained and ready to use!")
            st.markdown("</div>", unsafe_allow_html=True)
        else:
            st.markdown('<div class="warning-box">', unsafe_allow_html=True)
            st.markdown("⚠️ **Model Status**: Models need to be trained. Go to 'Model Performance' to train the models.")
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
        st.header("🏠 Property Valuation")
        # Note: Using "Gradient Boosting with Quantile Loss" instead of "Quantile Regression" 
        # because quantile regression typically refers to linear methods, while this app uses GradientBoostingRegressor with quantile loss
        
        if not MODEL_Q50_PATH.exists():
            st.error("⚠️ Models need to be trained first. Please go to 'Model Performance' to train the models.")
            return
        
        with st.form("valuation_form"):
            st.subheader("Enter Property Details")
            
            col1, col2 = st.columns(2)
            
            with col1:
                etage = st.number_input("Floor", min_value=1, max_value=20, value=2)
                age = st.number_input("Building Age (years)", min_value=0, max_value=100, value=15)
                aire_batiment = st.number_input("Building Area (m²)", min_value=20.0, max_value=1000.0, value=120.0, step=10.0)
            
            with col2:
                aire_lot = st.number_input("Lot Area (m²)", min_value=50.0, max_value=2000.0, value=300.0, step=50.0)
                prox_riverain = st.selectbox("Waterfront Proximity", options=[0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
            
            submitted = st.form_submit_button("Get Valuation")
            
            if submitted:
                try:
                    # Validate inputs
                    if aire_batiment <= 0:
                        st.error("Please enter a valid building area.")
                        return
                    
                    # Make prediction
                    low, median, high = predict_with_models(etage, age, aire_batiment, aire_lot, prox_riverain)
                    
                    # Get MAE for display
                    mae = get_model_mae()
                    
                    # Calculate ratio for KPI
                    ratio_result = find_nearest_neighbors_and_calculate_ratio(
                        etage, age, aire_batiment, aire_lot, prox_riverain, median
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
                    fig = create_shap_waterfall_chart(etage, age, aire_batiment, aire_lot, prox_riverain, median)
                    if fig:
                        st.plotly_chart(fig, use_container_width=True)
                        st.info("💡 **Price Construction**: This chart starts from the dataset's average property value and shows how each attribute contributes positively or negatively to the final predicted price.")
                    else:
                        st.warning("Could not generate the price construction chart.")
                    
                    # Additional metrics
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        price_per_m2 = median / aire_batiment
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
                    shap_results = compute_shap_values(etage, age, aire_batiment, aire_lot, prox_riverain)
                    
                    if shap_results:
                        st.info("💡 **SHAP Analysis**: Each value shows how much each feature contributes to the predicted price in dollars.")
                        
                        # Display SHAP values in organized sections
                        shap_sections = {
                            "Building Efficiency": ["Aire_Batiment"],
                            "Condition": ["Age"],
                            "Premium Location": ["Prox_Riverain"],
                            "Floor Level": ["Etage"],
                            "Lot Size": ["Aire_Lot"]
                        }
                        
                        for section_name, features in shap_sections.items():
                            st.markdown(f"#### {section_name}")
                            
                            for feature in features:
                                if feature in shap_results:
                                    result = shap_results[feature]
                                    shap_val = result["shap_value"]
                                    description = result["description"]
                                    
                                    # Color coding for positive/negative contributions
                                    if shap_val >= 0:
                                        color = "#2ca02c"  # Green for positive
                                        icon = "📈"
                                    else:
                                        color = "#d62728"  # Red for negative
                                        icon = "📉"
                                    
                                    col1, col2 = st.columns([1, 2])
                                    with col1:
                                        st.markdown(f"**{icon} ${shap_val:,.0f}**")
                                    with col2:
                                        st.markdown(f"*{description}*")
                                    
                                    # Add a small separator
                                    st.markdown("---")
                    else:
                        st.warning("SHAP analysis could not be computed for this property.")
                    
                except Exception as e:
                    st.error(f"An error occurred: {e}")
    

    
    elif page == "Model Performance":
        st.header("📈 Model Training & Performance")
        # Note: Using "Gradient Boosting with Quantile Loss" instead of "Quantile Regression" 
        # because quantile regression typically refers to linear methods, while this app uses GradientBoostingRegressor with quantile loss
        
        if st.button("🚀 Train Quantile Models", type="primary"):
            try:
                with st.spinner("Training models..."):
                    metrics = train_quantile_models(DATA_PATH)
                
                st.success("✅ Models trained and saved successfully!")
                
                # Display metrics
                st.metric("Mean Absolute Error", f"${metrics['MAE']:,.0f}")
                st.markdown("*Average prediction error*")
                
                # Model info
                st.subheader("Model Information")
                st.markdown("""
                **Gradient Boosting with Quantile Loss Models Trained:**
                - **5th percentile model**: Lower bound predictions
                - **50th percentile model**: Median predictions  
                - **95th percentile model**: Upper bound predictions
                
                **Features Used:**
                - Floor level, building age, building area, lot area, waterfront proximity
                - Raw property attributes from the dataset
                - Robust scaling for outlier handling
                """)
                
            except Exception as e:
                st.error(f"❌ Training failed: {e}")
        
        elif MODEL_Q50_PATH.exists():
            st.success("✅ Models are already trained and ready to use!")
            
            # Load and display model info
            try:
                data = joblib.load(MODEL_Q50_PATH)
                st.info("**Model Details:**")
                st.write(f"- Features: {len(data['features'])} raw attributes by valuation model")
                st.write("- Algorithm: Gradient Boosting with Quantile Loss")
                st.write("- Scaling: Robust Scaler")
            except:
                st.warning("Model files exist but may be corrupted. Please retrain.")
    
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
        The model is evaluated internally using several statistical metrics (including R²), but these are kept under the hood.  
        What you see in the app are the confidence intervals and error metrics (such as MAE), which are the most relevant for decision-making.
        """)

if __name__ == "__main__":
    main()
