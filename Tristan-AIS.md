**The Data & Model Pipeline**

**Appraiser's Information System is currently a Streamlit-based web app meant to whether confirm or invalidate your intuition about how much a property is worth in various areas in Quebec:**

Execution: Running train_models.py ingests the datasets, performs domain-specific feature scaling, trains the Gradient Boosting model, and serializes the state.

Handshake: The script exports a unified .joblib file containing the trained model structures.

Serving: At boot time, streamlit_real_estate_appraisal.py loads the compiled .joblib file into memory, providing instant, interactive client side predictions without retraining overhead.

**📊 Features Used**

The model leverages a highly curated matrix of physical, structural, and environmental features:

Floor(s): Total number of levels or specific floor location.

Age: Structural age of the building since construction.

Building Area (m²): Net livable or gross building surface area.

Lot Area (m²): Total land parcel area.

Waterfront Proximity: Binary or distance-based indicator for waterfront access.

Property Type: Categorical classification (Condo, Plex, Single-Family Home) optimized for heterogeneous appraisal sectors.

Park Proximity: Distance or routing score to the nearest public green space.

Metro Proximity: Spatial proximity metrics to public underground transit networks.

**📋 Tech Stack**

The entire ecosystem isolates its execution footprint inside a single, local virtual environment (.venv):

Python 3.9+

Streamlit (Web interface engine)

Scikit-learn (Gradient Boosting Regressor & Preprocessing frameworks)

Pandas & NumPy (Data manipulation matrix)

Plotly & Matplotlib (Interactive statistical visualizations)

Joblib (High-performance model persistence layer)

**🚀 Installation & Local Usage**

1. Clone the Repository

Bash
git clone https://github.com/tmayrand96/streamlit_real_estate_appraisal.git
cd streamlit_real_estate_appraisal

2. Configure the Virtual Environment (.venv)
Initialize your local isolated sandbox:

Bash
python -m venv .venv

* Activate on Windows (PowerShell): .venv\\Scripts\\Activate.ps1
* Activate on Mac/Linux: source .venv/bin/activate

3. Install Project Dependencies
Read the strict checklist manifest directly into your active environment:

Bash
pip install -r requirements.txt

4. Execute the App Modules
Step 1: Train and export the ML model artifact

Bash
python train_models.py

Step 2: Spin up the Streamlit interface

Bash
streamlit run streamlit_real_estate_appraisal.py

**📈 Model Evaluation & Observations**

The system tracks predictive integrity using standard regression metrics:

R² Score: Overall variance evaluation across designated sectors.

MAE: To flag absolute price divergence to the mean.

**Performance Observations**

Standard Market Properties: High precision and stable convergence within homogenous residential segments.

Luxury & Anomalous Segments: Predictions remain sensitive to high-cardinality architectural variables and volatile price distributions, marking an area for continuous optimization.

**🔮 Future Improvements**

Implementation of localized outlier-handling mechanisms to isolate premium properties.

Comparative benchmarking with Gradient Boosting Regressor.

Integration of continuous integration hooks and diff auditing workflows via custom GitHub CLI automated pipelines.

**📄 Author**
Tristan Mayrand
