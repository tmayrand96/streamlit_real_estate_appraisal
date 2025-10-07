# streamlit_real_estate_appraisal
Streamlit-based price estimation app for real estate

## Terminal utilities

In addition to the Streamlit interface you can manage the project entirely
from a shell by using the `real_estate_cli.py` helper script.

### Inspect the available regions

```bash
python real_estate_cli.py list-regions
```

### Generate predictions from the command line

You can supply either a CSV file or ad-hoc feature values. The example below
requests a prediction for the `BDF` region by specifying feature values
directly:

```bash
python real_estate_cli.py predict BDF \
  --values Etage=2 Age=15 Aire_Batiment=120 Aire_Lot=400 Prox_Riverain=0
```

To run predictions for multiple properties stored in a CSV file and save the
results to another CSV file:

```bash
python real_estate_cli.py predict PMR --from-csv my_properties.csv --output predictions.csv
```

### Download the trained `.joblib` models

Copy the joblib files for every region into a local directory:

```bash
python real_estate_cli.py download-models --destination ./exported-models
```

Add `--region REGION_KEY` (multiple times if needed) to only export specific
regions, and `--zip` to bundle each region's models into an archive instead of
individual files.
