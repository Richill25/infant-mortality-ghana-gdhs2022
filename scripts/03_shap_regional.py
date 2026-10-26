# ============================================================
# 03_SHAP AND REGIONAL ANALYSIS
# Infant Mortality in Ghana - GDHS 2022
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import joblib
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import mean_absolute_error, mean_squared_error
import warnings
warnings.filterwarnings('ignore')

# ── 1. LOAD MODELS AND DATA ──────────────────────────────────

xgb       = joblib.load('model_xgb.pkl')
rf        = joblib.load('model_rf.pkl')
lr        = joblib.load('model_lr.pkl')
X_test_ml = joblib.load('X_test_ml.pkl')
X_test_lr = joblib.load('X_test_lr.pkl')
y_test    = joblib.load('y_test.pkl')
X_ml      = joblib.load('X_ml.pkl')
X_lr      = joblib.load('X_lr_scaled.pkl')
y         = joblib.load('y_full.pkl')
df        = pd.read_csv('df_full_clean.csv')

print("Models and data loaded successfully.")
print(f"Test set shape : {X_test_ml.shape}")
print(f"Full ML shape  : {X_ml.shape}")

# ── READABLE LABELS ──────────────────────────────────────────
# Map DHS code-named new candidates to human-readable labels
# Confirmed from GHBR8CFL.DO variable labels
VAR_LABELS = {
    'v404' : 'currently_breastfeeding',
    'h22'  : 'had_fever_last_two_weeks',
    'h31'  : 'had_cough_last_two_weeks',
    'h42'  : 'given_iron_supplements',
    'h70a' : 'provider_measured_weight_3mo',
    'h70b' : 'provider_measured_height_3mo',
    'hw13' : 'result_height_weight_measurement',
    'v406' : 'currently_abstaining',
}

# Ghana DHS 2022 region codes → names
# These are LabelEncoder codes (alphabetical order), NOT the original V024 numeric codes
# Confirmed by running LabelEncoder().fit(region_names) on the 16 Ghana region strings
REGION_LABELS = {
    0 : 'Ahafo',
    1 : 'Ashanti',
    2 : 'Bono',
    3 : 'Bono East',
    4 : 'Central',
    5 : 'Eastern',
    6 : 'Greater Accra',
    7 : 'North East',
    8 : 'Northern',
    9 : 'Oti',
    10: 'Savannah',
    11: 'Upper East',
    12: 'Upper West',
    13: 'Volta',
    14: 'Western',
    15: 'Western North',
}

# Apply variable renaming to all feature matrices
X_test_ml = X_test_ml.rename(columns=VAR_LABELS)
X_test_lr = X_test_lr.rename(columns=VAR_LABELS)
X_ml      = X_ml.rename(columns=VAR_LABELS)
X_lr      = X_lr.rename(columns=VAR_LABELS)

# Apply region labels - region is stored as LabelEncoder integer in CSV
df['region'] = pd.to_numeric(df['region'], errors='coerce').map(REGION_LABELS).fillna(df['region'].astype(str))

print("    Variable labels and region names applied.")

# ── 2. SHAP - XGBOOST ────────────────────────────────────────

print("\nComputing SHAP values for XGBoost...")
explainer_xgb  = shap.TreeExplainer(xgb)
shap_values_xgb = explainer_xgb.shap_values(X_test_ml)

# Global bar plot - mean absolute SHAP
plt.figure()
shap.summary_plot(shap_values_xgb, X_test_ml,
                  plot_type='bar',
                  show=False)
plt.title('XGBoost - Global Feature Importance (SHAP)',
          fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('fig_shap_xgb_bar.png', dpi=150, bbox_inches='tight')
plt.show()

# Beeswarm plot - direction and magnitude
plt.figure()
shap.summary_plot(shap_values_xgb, X_test_ml,
                  plot_type='dot',
                  show=False)
plt.title('XGBoost - SHAP Beeswarm Plot',
          fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('fig_shap_xgb_beeswarm.png', dpi=150, bbox_inches='tight')
plt.show()

# ── 3. SHAP - RANDOM FOREST ──────────────────────────────────

print("\nComputing SHAP values for Random Forest...")
explainer_rf   = shap.TreeExplainer(rf)
shap_values_rf = explainer_rf.shap_values(X_test_ml)

# Handle RF shap output (may be 3D array)
if isinstance(shap_values_rf, list):
    sv_rf = shap_values_rf[1]
elif shap_values_rf.ndim == 3:
    sv_rf = shap_values_rf[:, :, 1]
else:
    sv_rf = shap_values_rf

# Global bar plot
plt.figure()
shap.summary_plot(sv_rf, X_test_ml,
                  plot_type='bar',
                  show=False)
plt.title('Random Forest - Global Feature Importance (SHAP)',
          fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('fig_shap_rf_bar.png', dpi=150, bbox_inches='tight')
plt.show()

# Beeswarm plot
plt.figure()
shap.summary_plot(sv_rf, X_test_ml,
                  plot_type='dot',
                  show=False)
plt.title('Random Forest - SHAP Beeswarm Plot',
          fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('fig_shap_rf_beeswarm.png', dpi=150, bbox_inches='tight')
plt.show()

# ── 4. SHAP - LOGISTIC REGRESSION ───────────────────────────
# Applied to the baseline statistical model to confirm whether
# SHAP-derived importance is consistent with the tree-based models.
# LinearExplainer uses the coefficient-feature product as SHAP values -
# the closest equivalent to LR odds ratios in the SHAP framework.

print("\nComputing SHAP values for Logistic Regression...")
# Use a sample of background data for the linear explainer
background_lr = shap.sample(X_test_lr, 100, random_state=42)
explainer_lr   = shap.LinearExplainer(lr, background_lr,
                                       feature_perturbation='interventional')
shap_values_lr = explainer_lr.shap_values(X_test_lr)

# Align LR column names with ML column names for comparison
# (LR has one-hot expanded columns; we use common original columns)
lr_cols_mapped = X_test_lr.columns.tolist()

# Global bar plot - LR
plt.figure()
shap.summary_plot(shap_values_lr, X_test_lr,
                  plot_type='bar', show=False)
plt.title('Logistic Regression - Global Feature Importance (SHAP)',
          fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('fig_shap_lr_bar.png', dpi=150, bbox_inches='tight')
plt.show()
print("    → fig_shap_lr_bar.png")

# Beeswarm plot - LR
plt.figure()
shap.summary_plot(shap_values_lr, X_test_lr,
                  plot_type='dot', show=False)
plt.title('Logistic Regression - SHAP Beeswarm Plot',
          fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('fig_shap_lr_beeswarm.png', dpi=150, bbox_inches='tight')
plt.show()
print("    → fig_shap_lr_beeswarm.png")


# ── 5. SHAP MEAN ABSOLUTE VALUES - ALL THREE MODELS ──────────

# XGBoost
xgb_shap_mean = pd.DataFrame({
    'Variable'  : X_test_ml.columns,
    'Mean_SHAP_XGB': np.abs(shap_values_xgb).mean(axis=0)
}).sort_values('Mean_SHAP_XGB', ascending=False)

# Random Forest
rf_shap_mean = pd.DataFrame({
    'Variable'  : X_test_ml.columns,
    'Mean_SHAP_RF': np.abs(sv_rf).mean(axis=0)
}).sort_values('Mean_SHAP_RF', ascending=False)

# Logistic Regression - aggregate one-hot columns back to original names
# For LR we take the mean absolute SHAP across all columns,
# then for one-hot expanded variables we sum contributions per original var
lr_shap_abs = pd.DataFrame({
    'Variable': X_test_lr.columns,
    'abs_shap': np.abs(shap_values_lr).mean(axis=0)
})
# Strip dummy suffixes (e.g. multiple_birth_1 → multiple_birth)
lr_shap_abs['base_var'] = lr_shap_abs['Variable'].str.replace(
    r'_[^_]+$', '', regex=True)
lr_shap_mean = (lr_shap_abs.groupby('base_var')['abs_shap']
                .sum()
                .reset_index()
                .rename(columns={'base_var':'Variable','abs_shap':'Mean_SHAP_LR'})
                .sort_values('Mean_SHAP_LR', ascending=False))

print('\nTop 10 by Mean |SHAP| - XGBoost')
print('='*45)
print(xgb_shap_mean.head(10).rename(
    columns={'Mean_SHAP_XGB':'Mean |SHAP|'}).to_string(index=False))

print('\nTop 10 by Mean |SHAP| - Random Forest')
print('='*45)
print(rf_shap_mean.head(10).rename(
    columns={'Mean_SHAP_RF':'Mean |SHAP|'}).to_string(index=False))

print('\nTop 10 by Mean |SHAP| - Logistic Regression')
print('='*45)
print(lr_shap_mean.head(10).rename(
    columns={'Mean_SHAP_LR':'Mean |SHAP|'}).to_string(index=False))

# ── Three-model SHAP comparison plot ─────────────────────────
# Merge all three on shared ML variable names
compare = (xgb_shap_mean
    .merge(rf_shap_mean,  on='Variable', how='outer')
    .merge(lr_shap_mean,  on='Variable', how='outer')
    .fillna(0)
    .sort_values('Mean_SHAP_XGB', ascending=False)
    .head(15))

fig, ax = plt.subplots(figsize=(13, 8))
x      = np.arange(len(compare))
width  = 0.28
bars1  = ax.bar(x - width,   compare['Mean_SHAP_XGB'], width,
                label='XGBoost',           color='#e67e22', edgecolor='white')
bars2  = ax.bar(x,            compare['Mean_SHAP_RF'],  width,
                label='Random Forest',     color='#2980b9', edgecolor='white')
bars3  = ax.bar(x + width,   compare['Mean_SHAP_LR'],  width,
                label='Logistic Regression', color='#27ae60', edgecolor='white')
ax.set_xticks(x)
ax.set_xticklabels(compare['Variable'], rotation=45, ha='right', fontsize=9)
ax.set_ylabel('Mean |SHAP Value|', fontsize=12)
ax.set_title('SHAP Feature Importance - Three-Model Comparison\n'
             'Consistency across XGBoost, Random Forest and Logistic Regression\n'
             '(GDHS 2022, bidx=1)',
             fontsize=13, fontweight='bold')
ax.legend(fontsize=11)
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('fig_shap_comparison.png', dpi=150, bbox_inches='tight')
plt.show()
print("    → fig_shap_comparison.png")

# ── 6. FEATURE IMPORTANCE (Built-in) ─────────────────────────

fig, axes = plt.subplots(1, 2, figsize=(16, 8))

rf_imp = pd.DataFrame({
    'Variable'  : X_ml.columns,
    'Importance': rf.feature_importances_
}).sort_values('Importance', ascending=True)

axes[0].barh(rf_imp['Variable'], rf_imp['Importance'],
             color='steelblue', edgecolor='white')
axes[0].set_title('Random Forest - Feature Importance',
                   fontsize=13, fontweight='bold')
axes[0].set_xlabel('Importance Score', fontsize=11)

xgb_imp = pd.DataFrame({
    'Variable'  : X_ml.columns,
    'Importance': xgb.feature_importances_
}).sort_values('Importance', ascending=True)

axes[1].barh(xgb_imp['Variable'], xgb_imp['Importance'],
             color='darkorange', edgecolor='white')
axes[1].set_title('XGBoost - Feature Importance',
                   fontsize=13, fontweight='bold')
axes[1].set_xlabel('Importance Score', fontsize=11)

plt.suptitle('Feature Importance Comparison (GDHS 2022)',
             fontsize=15, fontweight='bold')
plt.tight_layout()
plt.savefig('fig_feature_importance.png', dpi=150)
plt.show()

# ── 7. PROBABILITY CALIBRATION ───────────────────────────────

print("\nCalibrating XGBoost and Random Forest...")

calibrated_xgb = CalibratedClassifierCV(
    estimator=xgb, method='sigmoid', cv=5)
calibrated_xgb.fit(X_ml, y)

calibrated_rf = CalibratedClassifierCV(
    estimator=rf, method='sigmoid', cv=5)
calibrated_rf.fit(X_ml, y)

print("Calibration complete.")

# ── 8. REGIONAL ANALYSIS - RAW IMR ───────────────────────────

imr_raw = df.groupby('region')['infant_death'].agg(
    deaths='sum', births='count').reset_index()
imr_raw['IMR'] = (imr_raw['deaths'] / imr_raw['births'] * 1000).round(1)
imr_raw = imr_raw.sort_values('IMR', ascending=True)

print('\nRaw IMR by Region')
print('='*40)
print(imr_raw[['region','deaths','births','IMR']].to_string(index=False))

plt.figure(figsize=(10, 7))
plt.barh(imr_raw['region'], imr_raw['IMR'],
         color='steelblue', edgecolor='white')
plt.xlabel('IMR per 1,000 births', fontsize=12)
plt.title('Infant Mortality Rate by Region\n(Raw Data - GDHS 2022)',
          fontsize=14, fontweight='bold')
plt.axvline(x=25, color='red', linestyle='--',
            linewidth=1.5, label='SDG Target (25)')
plt.legend(fontsize=11)
plt.tight_layout()
plt.savefig('fig_regional_raw_imr.png', dpi=150)
plt.show()

# ── 9. REGIONAL ANALYSIS - PREDICTED RISK (XGBoost) ─────────

df_reg_xgb = df.copy()
df_reg_xgb['predicted_risk'] = calibrated_xgb.predict_proba(X_ml)[:, 1]

regional_xgb = df_reg_xgb.groupby('region').agg(
    mean_risk     = ('predicted_risk', 'mean'),
    actual_deaths = ('infant_death',   'sum'),
    total_births  = ('infant_death',   'count'),
    actual_IMR    = ('infant_death',
                     lambda x: round(x.mean() * 1000, 1))
).reset_index()

regional_xgb['mean_risk_per_1000'] = (
    regional_xgb['mean_risk'] * 1000).round(1)
regional_xgb = regional_xgb.sort_values(
    'mean_risk_per_1000', ascending=True)

print('\nXGBoost Predicted Regional Risk')
print('='*55)
print(regional_xgb[['region','actual_IMR',
                      'mean_risk_per_1000']].to_string(index=False))

plt.figure(figsize=(10, 7))
plt.barh(regional_xgb['region'],
         regional_xgb['mean_risk_per_1000'],
         color='darkorange', edgecolor='white')
plt.xlabel('Predicted Risk per 1,000 births', fontsize=12)
plt.title('XGBoost Predicted Infant Mortality Risk by Region\n(GDHS 2022)',
          fontsize=14, fontweight='bold')
plt.axvline(x=25, color='red', linestyle='--',
            linewidth=1.5, label='SDG Target (25)')
plt.legend(fontsize=11)
plt.tight_layout()
plt.savefig('fig_regional_xgb.png', dpi=150)
plt.show()

# ── 10. REGIONAL ANALYSIS - PREDICTED RISK (Random Forest) ────

df_reg_rf = df.copy()
df_reg_rf['predicted_risk'] = calibrated_rf.predict_proba(X_ml)[:, 1]

regional_rf = df_reg_rf.groupby('region').agg(
    mean_risk     = ('predicted_risk', 'mean'),
    actual_deaths = ('infant_death',   'sum'),
    total_births  = ('infant_death',   'count'),
    actual_IMR    = ('infant_death',
                     lambda x: round(x.mean() * 1000, 1))
).reset_index()

regional_rf['mean_risk_per_1000'] = (
    regional_rf['mean_risk'] * 1000).round(1)
regional_rf = regional_rf.sort_values(
    'mean_risk_per_1000', ascending=True)

print('\nRandom Forest Predicted Regional Risk')
print('='*55)
print(regional_rf[['region','actual_IMR',
                    'mean_risk_per_1000']].to_string(index=False))

plt.figure(figsize=(10, 7))
plt.barh(regional_rf['region'],
         regional_rf['mean_risk_per_1000'],
         color='steelblue', edgecolor='white')
plt.xlabel('Predicted Risk per 1,000 births', fontsize=12)
plt.title('Random Forest Predicted Infant Mortality Risk by Region\n(GDHS 2022)',
          fontsize=14, fontweight='bold')
plt.axvline(x=25, color='red', linestyle='--',
            linewidth=1.5, label='SDG Target (25)')
plt.legend(fontsize=11)
plt.tight_layout()
plt.savefig('fig_regional_rf.png', dpi=150)
plt.show()

# ── 11. REGIONAL ACCURACY METRICS ────────────────────────────

from scipy.stats import pearsonr

# XGBoost
mae_xgb  = mean_absolute_error(
    regional_xgb['actual_IMR'], regional_xgb['mean_risk_per_1000'])
rmse_xgb = np.sqrt(mean_squared_error(
    regional_xgb['actual_IMR'], regional_xgb['mean_risk_per_1000']))
corr_xgb, _ = pearsonr(
    regional_xgb['actual_IMR'], regional_xgb['mean_risk_per_1000'])

# Random Forest
mae_rf  = mean_absolute_error(
    regional_rf['actual_IMR'], regional_rf['mean_risk_per_1000'])
rmse_rf = np.sqrt(mean_squared_error(
    regional_rf['actual_IMR'], regional_rf['mean_risk_per_1000']))
corr_rf, _ = pearsonr(
    regional_rf['actual_IMR'], regional_rf['mean_risk_per_1000'])

print('\nRegional Prediction Accuracy')
print('='*50)
print(f"{'Model':<20} {'MAE':>8} {'RMSE':>8} {'Pearson r':>10}")
print(f"{'XGBoost':<20} {mae_xgb:>8.3f} {rmse_xgb:>8.3f} {corr_xgb:>10.3f}")
print(f"{'Random Forest':<20} {mae_rf:>8.3f} {rmse_rf:>8.3f} {corr_rf:>10.3f}")

# ── 12. PREDICTED VS ACTUAL SCATTER ──────────────────────────

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

for ax, name, reg_df, col in [
    (axes[0], 'XGBoost',       regional_xgb, 'darkorange'),
    (axes[1], 'Random Forest', regional_rf,  'steelblue')]:
    ax.scatter(reg_df['actual_IMR'],
               reg_df['mean_risk_per_1000'],
               color=col, s=80, edgecolors='black', linewidth=0.5)
    for _, row in reg_df.iterrows():
        ax.annotate(row['region'],
                    (row['actual_IMR'], row['mean_risk_per_1000']),
                    fontsize=7, ha='left')
    lims = [min(reg_df['actual_IMR'].min(),
                reg_df['mean_risk_per_1000'].min()) - 2,
            max(reg_df['actual_IMR'].max(),
                reg_df['mean_risk_per_1000'].max()) + 2]
    ax.plot(lims, lims, 'k--', linewidth=1, label='Perfect prediction')
    ax.set_xlabel('Actual IMR', fontsize=12)
    ax.set_ylabel('Predicted Risk', fontsize=12)
    ax.set_title(f'{name}\nActual vs Predicted Regional IMR',
                 fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('fig_regional_scatter.png', dpi=150)
plt.show()

# ── 13. REGIONAL SHAP DECOMPOSITION TABLE ────────────────────
# Computes mean SHAP values and mean feature values per region
# for the 7 consensus actionable predictors identified across all
# three models. Used to answer: which individual-level factors
# drive regional variation in predicted infant mortality risk?

print('\n── Regional SHAP Decomposition ─────────────────────────')

# Consensus actionable variables (those appearing in top 10 of
# both XGBoost and Random Forest and prominent in LR SHAP)
CONSENSUS_VARS = [
    'currently_breastfeeding',
    'provider_measured_height_3mo',
    'provider_measured_weight_3mo',
    'antenatal_visits',
    'given_iron_supplements',
    'preceding_birth_interval',
    'h43',
]

# Use a sample for SHAP speed (3,000 observations from full dataset)
np.random.seed(42)
n_sample = min(3000, len(X_ml))
sample_idx = np.random.choice(len(X_ml), n_sample, replace=False)
X_sample   = X_ml.iloc[sample_idx].reset_index(drop=True)
reg_sample = df['region'].iloc[sample_idx].reset_index(drop=True)

# Compute XGBoost SHAP on sample
print('Computing regional SHAP values (XGBoost, n=3,000 sample)...')
explainer_reg  = shap.TreeExplainer(xgb)
sv_reg         = explainer_reg.shap_values(X_sample)

# Build SHAP dataframe with region labels
sv_reg_df          = pd.DataFrame(sv_reg, columns=X_ml.columns)
sv_reg_df['region'] = reg_sample.values

# Mean SHAP per region for consensus vars (signed - direction matters)
regional_shap_mean = (sv_reg_df.groupby('region')[CONSENSUS_VARS]
                      .mean()
                      .round(4))

# Mean FEATURE VALUE per region for consensus vars
feat_df          = X_sample.copy()
feat_df['region'] = reg_sample.values
regional_feat_mean = (feat_df.groupby('region')[CONSENSUS_VARS]
                      .mean()
                      .round(3))

# Net SHAP contribution (sum across all 7 vars - more negative = more protected)
regional_shap_mean['net_shap'] = regional_shap_mean[CONSENSUS_VARS].sum(axis=1)
regional_shap_mean = regional_shap_mean.sort_values('net_shap', ascending=True)

# Print full table
print('\nMean SHAP values by region (XGBoost) - negative = protective')
print('='*95)
print(regional_shap_mean[CONSENSUS_VARS + ['net_shap']].to_string())

print('\nMean feature values by region')
print('='*95)
print(regional_feat_mean[CONSENSUS_VARS]
      .reindex(regional_shap_mean.index).to_string())

# ── Focused 3-variable table matching thesis Table 5.1 ───────
print('\n── Table for thesis (BF, ANC, Iron - SHAP + feature values) ──')
thesis_tbl = pd.DataFrame({
    'BF_rate'  : regional_feat_mean['currently_breastfeeding'],
    'SHAP_BF'  : regional_shap_mean['currently_breastfeeding'],
    'ANC_visits': regional_feat_mean['antenatal_visits'],
    'SHAP_ANC' : regional_shap_mean['antenatal_visits'],
    'Iron_supp' : regional_feat_mean['given_iron_supplements'],
    'SHAP_Iron' : regional_shap_mean['given_iron_supplements'],
}).reindex(regional_shap_mean.index).round(3)

print(thesis_tbl.to_string())

# ── Save as CSV ───────────────────────────────────────────────
regional_shap_mean.to_csv('regional_shap_decomposition.csv')
thesis_tbl.to_csv('regional_shap_table_thesis.csv')
print('\n  → regional_shap_decomposition.csv  (full 7-var SHAP table)')
print('  → regional_shap_table_thesis.csv   (3-var thesis summary table)')

# ── 14. REGIONAL SHAP BEESWARM - 3 KEY CONSENSUS VARIABLES ──
# Generates a grouped beeswarm showing SHAP value distributions
# for breastfeeding, antenatal visits, and iron supplementation
# separately for each region. This directly answers the research
# question: which individual-level factors drive regional variation?

print('\n── Regional SHAP Beeswarm (3 consensus variables) ──────')

PLOT_VARS = [
    'currently_breastfeeding',
    'antenatal_visits',
    'given_iron_supplements',
]

VAR_TITLES = {
    'currently_breastfeeding' : 'Currently Breastfeeding',
    'antenatal_visits'        : 'Antenatal Care Visits',
    'given_iron_supplements'  : 'Iron / Vitamin A Supplementation',
}

# Sort regions by breastfeeding SHAP (weakest to strongest protection)
region_order = (regional_shap_mean
                .sort_values('currently_breastfeeding', ascending=False)
                .index.tolist())

# Build long-form dataframe: one row per observation × variable
sv_plot = sv_reg_df[PLOT_VARS + ['region']].copy()
feat_plot = feat_df[PLOT_VARS + ['region']].copy()

fig, axes = plt.subplots(1, 3, figsize=(18, 9), sharey=True)

# colour palette - one colour per region (16 regions)
palette = plt.cm.tab20(np.linspace(0, 1, len(region_order)))
region_colour = {r: palette[i] for i, r in enumerate(region_order)}

for ax, var in zip(axes, PLOT_VARS):
    for reg in region_order:
        mask    = sv_plot['region'] == reg
        shap_v  = sv_plot.loc[mask, var].values
        feat_v  = feat_plot.loc[mask, var].values

        # jitter y position within each region band
        y_base  = region_order.index(reg)
        jitter  = np.random.uniform(-0.35, 0.35, size=len(shap_v))
        y_pos   = y_base + jitter

        sc = ax.scatter(
            shap_v, y_pos,
            c=feat_v,
            cmap='RdBu_r',
            vmin=np.percentile(feat_plot[var], 5),
            vmax=np.percentile(feat_plot[var], 95),
            alpha=0.4,
            s=6,
            linewidths=0,
        )

    ax.axvline(0, color='grey', linewidth=0.8, linestyle='--')
    ax.set_yticks(range(len(region_order)))
    ax.set_yticklabels(region_order, fontsize=9)
    ax.set_xlabel('SHAP Value\n(negative = protective)', fontsize=10)
    ax.set_title(VAR_TITLES[var], fontsize=11, fontweight='bold')
    ax.grid(axis='x', alpha=0.2)

    # Colourbar
    cbar = plt.colorbar(sc, ax=ax, pad=0.02, fraction=0.03)
    cbar.set_label('Feature value\n(red=high, blue=low)', fontsize=8)

axes[0].set_ylabel('Region (sorted by breastfeeding SHAP,\nweakest to strongest protection)',
                   fontsize=10)

fig.suptitle(
    'Regional SHAP Decomposition - Three Key Consensus Predictors\n'
    'XGBoost | GDHS 2022 | Each dot = one birth observation\n'
    'Negative SHAP = protective effect; closer to zero = weaker protection',
    fontsize=12, fontweight='bold', y=1.01
)
plt.tight_layout()
plt.savefig('fig_shap_regional_beeswarm.png', dpi=150,
            bbox_inches='tight')
plt.show()
print('  → fig_shap_regional_beeswarm.png')
print('''
  SHAP outputs:
    fig_shap_lr_bar.png           ← Logistic Regression SHAP (baseline)
    fig_shap_lr_beeswarm.png
    fig_shap_xgb_bar.png          ← XGBoost SHAP
    fig_shap_xgb_beeswarm.png
    fig_shap_rf_bar.png           ← Random Forest SHAP
    fig_shap_rf_beeswarm.png
    fig_shap_comparison.png       ← Three-model comparison

  Regional outputs:
    fig_regional_raw_imr.png
    fig_regional_xgb.png
    fig_regional_rf.png
    fig_regional_scatter.png
    fig_feature_importance.png
    regional_shap_decomposition.csv   ← NEW: full regional SHAP table
    regional_shap_table_thesis.csv    ← NEW: thesis summary table
''')
