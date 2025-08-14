# 🏠 Property Valuation System - Quantile Regression

An advanced property valuation system using quantile regression to provide accurate property valuations with confidence intervals.

## 🌟 Features

- **Quantile Regression**: Provides 5th, 50th, and 95th percentile predictions
- **Confidence Intervals**: Understand uncertainty in property valuations
- **Advanced Feature Engineering**: Engineered features for better accuracy
- **Beautiful UI**: Professional, user-friendly web interface
- **Interactive Visualizations**: Charts and graphs for better insights

## 🚀 Live Demo

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://your-app-name.streamlit.app)

## 📋 Requirements

- Python 3.8+
- Streamlit
- Scikit-learn
- Pandas
- NumPy
- Plotly
- Joblib

## 🛠️ Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/property-valuation-system.git
cd property-valuation-system
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the application:
```bash
streamlit run streamlit_property_valuation_quantile.py
```

## 📁 Project Structure

```
property-valuation-system/
├── streamlit_property_valuation_quantile.py  # Main application
├── donnees_BDF.csv                          # Training dataset
├── requirements.txt                          # Python dependencies
├── .streamlit/config.toml                   # Streamlit configuration
├── README.md                                # This file
└── .gitignore                               # Git ignore file
```

## 🎯 Usage

1. **Train the Model**: Go to "Model Performance" and click "Train Quantile Models"
2. **Get Valuations**: Use "Property Valuation" for custom estimates
3. **View Examples**: Check "Demo Properties" for sample valuations

## 📊 Model Information

- **Algorithm**: Gradient Boosting with Quantile Loss
- **Features**: Building area, terrace area, age, bedrooms, pool area
- **Engineered Features**: Ratios, polynomials, interactions
- **Scaling**: Robust Scaler for outlier handling

## 🔧 Deployment on Streamlit Cloud

1. Push your code to GitHub
2. Connect your repository to [Streamlit Cloud](https://streamlit.io/cloud)
3. Deploy with the following settings:
   - **Main file path**: `streamlit_property_valuation_quantile.py`
   - **Python version**: 3.9 or higher

## 📈 Performance

- **R² Score**: High accuracy with quantile regression
- **Confidence Intervals**: 5th to 95th percentile predictions
- **Feature Engineering**: Advanced feature creation for better predictions

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 📞 Contact

For questions or support, please open an issue on GitHub. 