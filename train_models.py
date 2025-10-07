# train_models.py
from pathlib import Path
import joblib, pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, RobustScaler
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error

# ----- import your config & helpers from the app -----
from streamlit_real_estate_appraisal import REGION_CONFIG, CANON_TARGET
from streamlit_real_estate_appraisal import create_features, model_path_for

BASE_DIR   = Path(__file__).parent
DATA_DIR   = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
MODELS_DIR.mkdir(exist_ok=True)

def load_csv(region_key: str) -> pd.DataFrame:
    cfg = REGION_CONFIG[region_key]
    df = pd.read_csv(cfg["data_path"], encoding="utf-8-sig")
    # normalize common target variants; keep this minimal and explicit
    if CANON_TARGET not in df.columns:
        for alt in ["Prix_de_Vente", "prix_de_vente", "Prix", "Price", "price_sold"]:
            if alt in df.columns:
                df = df.rename(columns={alt: CANON_TARGET})
                break
    if CANON_TARGET not in df.columns:
        raise ValueError(f"[{cfg['name']}] target '{CANON_TARGET}' not found. Columns: {list(df.columns)}")
    return df

def train_region(region_key: str):
    cfg = REGION_CONFIG[region_key]
    df = load_csv(region_key)
    # IMPORTANT: keep the target; only restrict to features at the X/y slice
    df = create_features(df, region_key, is_training=True)
    if CANON_TARGET not in df.columns:
        raise ValueError(f"[{cfg['name']}] target disappeared after feature prep.")

    feat = [c for c in cfg["feature_cols"] if c in df.columns]
    if not feat:
        raise ValueError(f"[{cfg['name']}] no usable features. Columns: {list(df.columns)}")

    X = df[feat].copy()
    y = df[CANON_TARGET].astype(float)
    num_cols = [c for c in cfg["num_cols"] if c in feat]
    cat_cols = [c for c in cfg["cat_cols"] if c in feat]

    numeric = Pipeline([('imputer', SimpleImputer(strategy='median')),
                        ('scaler', RobustScaler())])
    if cat_cols:
        categorical = Pipeline([('imputer', SimpleImputer(strategy='constant', fill_value='__missing__')),
                                ('onehot', OneHotEncoder(handle_unknown='ignore', drop='first'))])
        preproc = ColumnTransformer([('num', numeric, num_cols),
                                     ('cat', categorical, cat_cols)],
                                    remainder='drop')
    else:
        preproc = ColumnTransformer([('num', numeric, num_cols)], remainder='drop')

    metrics = {}

    for alpha in (0.05, 0.50, 0.95):
        pipe = Pipeline([('preproc', preproc),
                         ('model', GradientBoostingRegressor(loss="quantile", alpha=alpha, random_state=42))])

        X_tr, X_va, y_tr, y_va = train_test_split(X, y, test_size=0.2, random_state=42)
        pipe.fit(X_tr, y_tr)

        model_blob = {"pipeline": pipe, "features": feat, "region": region_key,
                      "num_cols": num_cols, "cat_cols": cat_cols}

        if alpha == 0.50:
            y_hat = pipe.predict(X_va)
            metrics["R2"] = r2_score(y_va, y_hat)
            metrics["MAE"] = mean_absolute_error(y_va, y_hat)
            model_blob["mae"] = metrics["MAE"]

        out = model_path_for(region_key, alpha)
        out.parent.mkdir(exist_ok=True, parents=True)
        joblib.dump(model_blob, out)
        print(f"[{cfg['name']}] wrote {out.name}")

    print(f"[{cfg['name']}] metrics: {metrics}")
    return metrics

if __name__ == "__main__":
    for rk in ("BDF", "PMR", "Ste-Rose"):
        try:
            print(f"== Training {rk} ==")
            train_region(rk)
        except Exception as e:
            print(f"!! {rk} failed: {e}")
