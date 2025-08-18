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
            joblib.dump({"model": gbr, "scaler": scaler, "features": feature_cols}, path)
            models[alpha] = gbr
            if alpha == 0.50:
                y_pred_val = gbr.predict(X_val)
                metrics["R2"] = r2_score(y_val, y_pred_val)
                metrics["MAE"] = mean_absolute_error(y_val, y_pred_val)

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

def create_quantile_chart(low, median, high):
    """Create a beautiful chart showing the quantile predictions"""
    fig = go.Figure()
    
    # Add bar for median prediction
    fig.add_trace(go.Bar(
        x=['Median Prediction'],
        y=[median],
        name='Median (50th percentile)',
        marker_color='#1f77b4',
        width=0.6
    ))
    
    # Add error bars for quantile range
    fig.add_trace(go.Scatter(
        x=['Median Prediction'],
        y=[high],
        mode='markers',
        name='Best-case Market Value (95th percentile)',
        marker=dict(color='#ff7f0e', size=10, symbol='triangle-up'),
        showlegend=True
    ))
    
    fig.add_trace(go.Scatter(
        x=['Median Prediction'],
        y=[low],
        mode='markers',
        name='Lower Bound (5th percentile)',
        marker=dict(color='#2ca02c', size=10, symbol='triangle-down'),
        showlegend=True
    ))
    
    # Add range line
    fig.add_trace(go.Scatter(
        x=['Median Prediction', 'Median Prediction'],
        y=[low, high],
        mode='lines',
        name='Prediction Range',
        line=dict(color='#d62728', width=3, dash='dash'),
        showlegend=True
    ))
    
    fig.update_layout(
        title="Property Value Prediction with Confidence Intervals",
        yaxis_title="Predicted Price ($)",
        height=400,
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    return fig

def create_demo_properties_chart():
    """Create chart for demo properties"""
    demo_props = [
        {"name": "Small Apartment", "etage": 2, "age": 5, "aire_batiment": 80, "aire_lot": 200, "prox_riverain": 0},
        {"name": "Family House", "etage": 1, "age": 20, "aire_batiment": 150, "aire_lot": 500, "prox_riverain": 1},
        {"name": "Studio", "etage": 3, "age": 70, "aire_batiment": 35, "aire_lot": 100, "prox_riverain": 0},
        {"name": "Luxury Villa", "etage": 1, "age": 10, "aire_batiment": 300, "aire_lot": 1000, "prox_riverain": 1}
    ]
    
    results = []
    for prop in demo_props:
        try:
            low, median, high = predict_with_models(
                prop["etage"], prop["age"], prop["aire_batiment"], 
                prop["aire_lot"], prop["prox_riverain"]
            )
            price_per_m2 = median / prop["aire_batiment"]
            
            results.append({
                "Property": prop["name"],
                "Median Price": median,
                "Price Range": f"${low:,.0f} -   ${high:,.0f}",
                "Price per m²": price_per_m2,
                "Building Area": prop["aire_batiment"],
                "Age": prop["age"],
                "Floor": prop["etage"]
            })
        except:
            # Skip if model not trained
            continue
    
    if results:
        df_results = pd.DataFrame(results)
        
        # Create price comparison chart
        fig = px.bar(df_results, x="Property", y="Median Price",
                     title="Estimated Property Values (Demo Properties)",
                     color="Bedrooms",
                     color_continuous_scale="viridis")
        
        fig.update_layout(height=400)
        
        return fig, df_results
    
    return None, None

def main():
    # Header
    st.markdown('<h1 class="main-header">🏠 Property Valuation System</h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; font-size: 1.2rem; color: #666;">Advanced Quantile Regression for Accurate Property Valuations</p>', unsafe_allow_html=True)
    
    # Sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox(
        "Choose a page:",
        ["Home", "Property Valuation", "Demo Properties", "Model Performance", "About"]
    )
    
    if page == "Home":
        st.markdown("""
        ## Welcome to the Advanced Property Valuation System
        
        This system uses **quantile regression** to provide not just point estimates, but also confidence intervals for property valuations.
        
        ### Key Features:
        - **Quantile Regression**: Provides 5th, 50th, and 95th percentile predictions
        - **Confidence Intervals**: Understand the uncertainty in your valuations
        - **Advanced Features**: Engineered features for better accuracy
        - **Interactive Interface**: Beautiful, user-friendly web interface
        
        ### How to Use:
        1. **Train the Model**: Go to "Model Performance" to train the quantile models
        2. **Get Valuations**: Use "Property Valuation" for custom estimates
        3. **View Examples**: Check "Demo Properties" for sample valuations
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
            st.metric("Model Type", "Quantile Regression", "Advanced")
        
        with col2:
            st.metric("Prediction Range", "5th - 95th percentile", "Confidence Intervals")
        
        with col3:
            st.metric("Features", "5 Raw Attributes", "High Accuracy")
    
    elif page == "Property Valuation":
        st.header("🏠 Property Valuation")
        
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
                    
                    # Display results
                    st.markdown('<div class="prediction-card">', unsafe_allow_html=True)
                    st.markdown(f"## Estimated Property Value")
                    st.markdown(f"# ${median:,.0f}")
                    st.markdown(f"*Confidence Range: ${low:,.0f} - ${high:,.0f}*")
                    st.markdown("</div>", unsafe_allow_html=True)
                    
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
                        st.markdown("### Best-case Market Value")
                        st.markdown(f"## $ {high:,.0f}")
                        st.markdown("*95th percentile*")
                        st.markdown("</div>", unsafe_allow_html=True)
                    
                    # Chart
                    fig = create_quantile_chart(low, median, high)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Additional metrics
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        price_per_m2 = median / aire_batiment
                        st.metric("Price per m²", f"${price_per_m2:,.0f}")
                    
                    with col2:
                        confidence_range = high - low
                        st.metric("Confidence Range", f"${confidence_range:,.0f}")
                    
                    with col3:
                        confidence_percentage = ((high - low) / median) * 100
                        st.metric("Uncertainty", f"{confidence_percentage:.1f}%")
                    
                    # Property analysis
                    st.subheader("Property Analysis")
                    
                    analysis_text = []
                    analysis_text.append(f"**Building Efficiency**: {aire_batiment} m² of living space")
                    
                    if age < 10:
                        analysis_text.append("**Condition**: New building (excellent condition)")
                    elif age < 30:
                        analysis_text.append("**Condition**: Modern building (good condition)")
                    elif age < 50:
                        analysis_text.append("**Condition**: Standard building (fair condition)")
                    else:
                        analysis_text.append("**Condition**: Older building (may need renovation)")
                    
                    if prox_riverain == 1:
                        analysis_text.append("**Premium Location**: Waterfront property")
                    
                    analysis_text.append(f"**Floor Level**: {etage} floor(s)")
                    analysis_text.append(f"**Lot Size**: {aire_lot} m² total area")
                    
                    for text in analysis_text:
                        st.markdown(f"• {text}")
                    
                except Exception as e:
                    st.error(f"An error occurred: {e}")
    
    elif page == "Demo Properties":
        st.header("📊 Demo Properties")
        
        if not MODEL_Q50_PATH.exists():
            st.error("⚠️ Models need to be trained first. Please go to 'Model Performance' to train the models.")
            return
        
        fig, df_results = create_demo_properties_chart()
        
        if fig is not None:
            # Display chart
            st.plotly_chart(fig, use_container_width=True)
            
            # Display detailed results
            st.subheader("Detailed Results")
            st.dataframe(df_results, use_container_width=True)
            
            # Additional insights
            st.subheader("Key Insights")
            col1, col2 = st.columns(2)
            
            with col1:
                st.info("**Size Matters**: Larger properties generally have better price-to-area ratios.")
            
            with col2:
                st.info("**Age Impact**: Newer properties command premium prices, but well-maintained older properties can still be valuable.")
        else:
            st.warning("No demo results available. Please train the model first.")
    
    elif page == "Model Performance":
        st.header("📈 Model Training & Performance")
        
        if st.button("🚀 Train Quantile Models", type="primary"):
            try:
                with st.spinner("Training models..."):
                    metrics = train_quantile_models(DATA_PATH)
                
                st.success("✅ Models trained and saved successfully!")
                
                # Display metrics
                col1, col2 = st.columns(2)
                
                with col1:
                    st.metric("R² Score", f"{metrics['R2']:.4f}")
                    st.markdown(f"*The model explains {metrics['R2']*100:.1f}% of price variations*")
                
                with col2:
                    st.metric("Mean Absolute Error", f"${metrics['MAE']:,.0f}")
                    st.markdown("*Average prediction error*")
                
                # Model info
                st.subheader("Model Information")
                st.markdown("""
                **Quantile Regression Models Trained:**
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
                st.write(f"- Features: {len(data['features'])} engineered features")
                st.write("- Algorithm: Gradient Boosting with Quantile Loss")
                st.write("- Scaling: Robust Scaler")
                
                st.button("🔄 Retrain Models", type="secondary")
            except:
                st.warning("Model files exist but may be corrupted. Please retrain.")
    
    elif page == "About":
        st.header("ℹ️ About")
        
        st.markdown("""
        ## Advanced Property Valuation System
        
        This system uses **quantile regression** to provide comprehensive property valuations with confidence intervals.
        
        ### Technology Stack:
        - **Python**: Core programming language
        - **Streamlit**: Web application framework
        - **Scikit-learn**: Machine learning library with quantile regression
        - **Plotly**: Interactive visualizations
        - **Pandas & NumPy**: Data manipulation
        
        ### Model Architecture:
        The system uses **quantile regression** with Gradient Boosting:
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
        
        ### Advantages of Quantile Regression:
        - **Uncertainty Quantification**: Provides confidence intervals
        - **Robust Predictions**: Less sensitive to outliers
        - **Risk Assessment**: Helps understand prediction uncertainty
        - **Better Decision Making**: Range of possible values instead of single point
        
        ### Data Source:
        The model is trained on real estate data from Statistics Canada.
        """)

if __name__ == "__main__":
    main()
