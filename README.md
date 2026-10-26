# Infant mortality risk in Ghana: analysis code

Analysis code for the manuscript "Statistical and machine learning methods for analyzing infant mortality risk in Ghana: a secondary analysis of the 2022 Demographic and Health Survey" by Richill Ataa Nyantakyiwaa and Fabrizio Ruggeri (under review).

## What this repository contains

The code trains and evaluates infant mortality risk models (logistic regression, Random Forest, XGBoost and a soft-voting ensemble) on births from the 2022 Ghana Demographic and Health Survey (GDHS), explains them with SHAP values, summarises risk by region, and fits an ARIMA model to the national annual infant mortality series.

**Sample.** 10,356 most recent births (bidx = 1), with 237 infant deaths. The infant mortality rate in this restricted sample is 22.9 per 1,000 live births, compared with 39.1 per 1,000 in the full birth history. The difference is a selection effect of the restriction, because younger, lower-parity mothers are over-represented, so the 22.9 figure should not be read as a national rate.

## Data

No survey data are included in this repository. The GDHS data are free but must be requested from the DHS Program (https://dhsprogram.com/data/), and the DHS terms do not allow the files, or datasets derived from them, to be redistributed. You need the 2022 Ghana Births Recode file, `GHBR8CFL` (Stata format, `GHBR8CFL.DTA`).

## Repository contents

| Folder | Contents |
|--------|----------|
| `scripts/` | The four analysis scripts |
| `notebooks/` | Master notebook that runs the scripts in order (saved outputs cleared) |
| `figures/` | Figures produced by the scripts |
| `results/` | Regional SHAP tables, aggregated to Ghana's 16 regions |
| `data/` | Empty by design, see `data/README.md` |

## Running the analysis

1. Install Python 3.9 or later and the packages in `requirements.txt`.
2. Create a working folder, copy the four scripts into it, and place `GHBR8CFL.DTA` in the same folder.
3. Run the scripts in order from that folder:

```bash
python 01_eda_feature_selection.py
python 02_machine_learning.py
python 03_shap_regional.py
python 04_national_arimax.py
```

Script 01 writes `df_full_clean.csv`, which scripts 02 and 03 read. Figures, regional tables and model files are written to the folder where you run the scripts. Random seeds are fixed (`random_state = 42`). The notebook runs the scripts with `%run`, so it must also be placed in the working folder.

Software versions used to produce the results reported in the manuscript (scripts 02 and 03): Python 3.12.3, pandas 3.0.2, NumPy 2.4.4, scikit-learn 1.8.0, XGBoost 3.4.1, SHAP 0.54.1, SciPy 1.17.1. Small differences in results are expected with other library versions.

## Scripts

| Script | What it does |
|--------|--------------|
| `01_eda_feature_selection.py` | Cleaning, missingness handling, exploratory analysis, and three-stage feature selection (mutual information, PCA, BIC forward stepwise logistic regression). Writes `df_full_clean.csv` |
| `02_machine_learning.py` | Logistic regression, Random Forest, XGBoost and a soft-voting ensemble with class weighting (no SMOTE). Per-model F1-optimal thresholds, stratified five-fold and geographic (cluster-grouped) cross-validation, calibration, and saved models |
| `03_shap_regional.py` | SHAP explanations for all three models, Platt-calibrated regional risk, and the regional SHAP decomposition |
| `04_national_arimax.py` | National annual series for 2000 to 2022 (2001 excluded because of a small birth cohort, n = 540, leaving 22 annual observations). ARIMA order chosen by ACF and PACF inspection with a parsimony check, forecast to 2030, and regional comparison with the national fitted values |

## Implementation notes

- The enumeration-area cluster and strata identifiers (`cluster_id`, `strata_id`) stay in `df_full_clean.csv`. Script 02 uses `cluster_id` only as the grouping variable for geographic cross-validation and excludes both identifiers from the model inputs.
- Survey weights (`v005`) are dropped in script 01, so the weighting branch in script 02 is inactive and models are fitted without survey weights.
- The ARIMAX test with a machine-learning-derived exogenous series and the rank correlation between the machine-learning and ARIMA regional orderings reported in the manuscript are not included in these scripts.

## Citation

See `CITATION.cff`. A Zenodo DOI will be added after the first release.

## Licence

MIT, see `LICENSE`.

## Contact

Richill Ataa Nyantakyiwaa, richill.nyantakyiwaa@aims.ac.rw
