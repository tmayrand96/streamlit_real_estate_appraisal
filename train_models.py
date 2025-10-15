# ==================== train_models.py (standalone) ====================
from pathlib import Path
import pandas as pd
import numpy as np
import joblib
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, RobustScaler
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error

# ---- Paths ----
BASE_DIR   = Path(__file__).parent
DATA_DIR   = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
MODELS_DIR.mkdir(exist_ok=True)

# ---- Canonical target ----
CANON_TARGET = "Prix_de_vente"

# ---- Region configs (match your CSV headers exactly) ----
REGION_CONFIG = {
    "BDF": {
        "name": "Bois-Des-Filion",
        "data_path": DATA_DIR / "donnees_BDF.csv",
        "feature_cols": ["Etage", "Age", "Aire_Batiment", "Aire_Lot", "Prox_Riverain"],
        "num_cols":     ["Etage", "Age", "Aire_Batiment", "Aire_Lot", "Prox_Riverain"],
        "cat_cols":     [],
        "model_prefix": "bdf",
    },
    "PMR": {
        "name": "Plateau Mont-Royal",
        "data_path": DATA_DIR / "Dataset_PMR.csv",
        "feature_cols": ["CONDO", "5PLEX_ET_MOINS", "6PLEX_ET_PLUS", "UNIFAMILIALE", "ETAGES", "AGE", "AIRE_HABITABLE", "TAXES_AN", "Prox_Parc", "Prox_Metro"],
        "num_cols":     ["CONDO", "5PLEX_ET_MOINS", "6PLEX_ET_PLUS", "UNIFAMILIALE", "ETAGES", "AGE", "AIRE_HABITABLE", "TAXES_AN", "Prox_Parc", "Prox_Metro"],
        "cat_cols":     [],
        "model_prefix": "pmr",
    },
    "Ste-Rose": {
        "name": "Sainte-Rose",
        "data_path": DATA_DIR / "Dataset_Ste-Rose.csv",
        "feature_cols": ["Etage", "Age", "Aire_Batiment_m2", "Aire_Lot_m2", "Garage", "Amenagement_paysager"],
        "num_cols":     ["Etage", "Age", "Aire_Batiment_m2", "Aire_Lot_m2", "Garage", "Amenagement_paysager"],
        "cat_cols":     [],
        "model_prefix": "ste_rose",
    },
}

# ---- Helpers ----
def model_path_for(region_key: str, alpha: float) -> Path:
    q = {0.05: "q05", 0.50: "q50", 0.95: "q95"}[alpha]
    return MODELS_DIR / f"{REGION_CONFIG[region_key]['model_prefix']}_model_{q}.joblib"

def load_csv(region_key: str) -> pd.DataFrame:
    cfg = REGION_CONFIG[region_key]
    df = pd.read_csv(cfg["data_path"], encoding="utf-8-sig")
    # Normalize trivial header differences
    df.columns = [c.strip() for c in df.columns]
    # Minimal target normalization
    if CANON_TARGET not in df.columns:
        for alt in ["Prix_de_Vente", "prix_de_vente", "Prix", "Price", "price_sold"]:
            if alt in df.columns:
                df = df.rename(columns={alt: CANON_TARGET})
                break
    if CANON_TARGET not in df.columns:
        raise ValueError(f"[{cfg['name']}] Target '{CANON_TARGET}' not found. Columns: {list(df.columns)}")
    return df

def coerce_types(df: pd.DataFrame, region_key: str, features: list) -> pd.DataFrame:
    """Coerce numeric/boolean/categorical types in-place for clean modeling."""
    cfg = REGION_CONFIG[region_key]
    num_cols = [c for c in cfg["num_cols"] if c in features]
    cat_cols = [c for c in cfg["cat_cols"] if c in features]

    # Coerce numeric-like
    for c in num_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    return df

def train_region(region_key: str):
    cfg = REGION_CONFIG[region_key]
    print(f"== Training {region_key} ==")
    df = load_csv(region_key)

    # IMPORTANT: keep the target; do not slice it out
    features = [c for c in cfg["feature_cols"] if c in df.columns]
    missing  = [c for c in cfg["feature_cols"] if c not in df.columns]
    if missing:
        raise ValueError(f"[{cfg['name']}] Missing feature columns: {missing}. Found: {list(df.columns)}")

    # Coerce types for modeling
    df = coerce_types(df, region_key, features)

    # Drop rows with missing target; cast y to float
    df = df.dropna(subset=[CANON_TARGET])
    y  = df[CANON_TARGET].astype(float)
    X  = df[features].copy()

    # (Optional) If any feature column ends up entirely NaN after coercion, drop it with a clear message
    all_nan = [c for c in features if df[c].isna().all()]
    if all_nan:
        print(f"[{cfg['name']}] Dropping all-NaN features: {all_nan}")
        features = [c for c in features if c not in all_nan]
        X = df[features].copy()

    num_cols = [c for c in cfg["num_cols"] if c in features]
    cat_cols = [c for c in cfg["cat_cols"] if c in features]

    numeric = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  RobustScaler()),
    ])

    transformers = [("num", numeric, num_cols)]
    if cat_cols:
        categorical = Pipeline([
            ("imputer", SimpleImputer(strategy="constant", fill_value="__missing__")),
            ("onehot",  OneHotEncoder(handle_unknown="ignore", drop="first")),
        ])
        transformers.append(("cat", categorical, cat_cols))

    preproc = ColumnTransformer(transformers, remainder="drop")

    metrics = {}
    for alpha in (0.05, 0.50, 0.95):
        pipe = Pipeline([
            ("preproc", preproc),
            ("model", GradientBoostingRegressor(loss="quantile", alpha=alpha, random_state=42)),
        ])
        X_tr, X_va, y_tr, y_va = train_test_split(X, y, test_size=0.2, random_state=42)
        pipe.fit(X_tr, y_tr)

        blob = {
            "pipeline":  pipe,
            "features":  features,
            "region":    region_key,
            "num_cols":  num_cols,
            "cat_cols":  cat_cols,
        }

        if alpha == 0.50:
            y_hat = pipe.predict(X_va)
            metrics["R2"]  = r2_score(y_va, y_hat)
            metrics["MAE"] = mean_absolute_error(y_va, y_hat)
            blob["mae"]    = metrics["MAE"]

        out = model_path_for(region_key, alpha)
        out.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(blob, out)
        print(f"  -> wrote {out.name}")

    print(f"[{cfg['name']}] metrics: {metrics}")
    return metrics

if __name__ == "__main__":
    for rk in ("BDF", "PMR", "Ste-Rose"):
        try:
            train_region(rk)
        except Exception as e:
            print(f"!! {rk} failed: {e}")
# ==================== end ====================