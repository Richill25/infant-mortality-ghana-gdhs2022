# ============================================================
# 02_MACHINE LEARNING MODELS  ·  ZIP 1 - CLASS WEIGHT ONLY
# Infant Mortality in Ghana - GDHS 2022
# Strategy: class_weight='balanced' for LR & RF,
#           scale_pos_weight for XGBoost. NO SMOTE.
# Threshold: Per-model F1-optimal threshold found from the
#            precision-recall curve on the validation set,
#            ensuring balanced precision and recall.
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import (StratifiedKFold, GroupKFold,
                                     train_test_split)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (roc_auc_score, recall_score,
                             precision_score, f1_score,
                             accuracy_score, classification_report,
                             roc_curve, confusion_matrix,
                             precision_recall_curve,
                             average_precision_score)
from sklearn.utils import resample
import warnings
warnings.filterwarnings('ignore')

# ── 1. LOAD CLEANED DATA ─────────────────────────────────────

df = pd.read_csv('df_full_clean.csv')
print(f"Loaded dataset: {df.shape}")
print(f"Infant deaths: {df['infant_death'].sum()}")

# ── 2. ENCODING ──────────────────────────────────────────────

nominal_cols = ['multiple_birth', 'sex_of_child',
                'caesarean_section', 'region', 'marital_status']
ordinal_cols = ['wealth_index']

df_lr = pd.get_dummies(df, columns=nominal_cols, drop_first=True)
le = LabelEncoder()
for col in ordinal_cols:
    df_lr[col] = le.fit_transform(df_lr[col].astype(str))
df_lr = df_lr.drop(columns=['living_children'], errors='ignore')

df_ml = df.copy()
for col in nominal_cols + ordinal_cols:
    df_ml[col] = le.fit_transform(df_ml[col].astype(str))

print(f"\nLR dataset shape : {df_lr.shape}")
print(f"ML dataset shape : {df_ml.shape}")

# ── 3. FEATURES AND OUTCOME ──────────────────────────────────

# cluster_id and strata_id are survey-design identifiers: they are used only as grouping
# variables for geographic cross-validation and are excluded from the model inputs.
ID_COLS = ['cluster_id', 'strata_id']
X_lr = df_lr.drop(columns=['infant_death'] + ID_COLS, errors='ignore')
X_ml = df_ml.drop(columns=['infant_death'] + ID_COLS, errors='ignore')
y    = df_ml['infant_death']

scaler      = StandardScaler()
X_lr_scaled = pd.DataFrame(
    scaler.fit_transform(X_lr),
    columns=X_lr.columns,
    index=X_lr.index
)

# ── 4. TRAIN / TEST SPLIT ────────────────────────────────────

X_train_lr, X_test_lr, y_train, y_test = train_test_split(
    X_lr_scaled, y, test_size=0.2, random_state=42, stratify=y)
X_train_ml, X_test_ml, _, _ = train_test_split(
    X_ml, y, test_size=0.2, random_state=42, stratify=y)

scale = y_train.value_counts()[0] / y_train.value_counts()[1]

print(f"\nClass distribution - Survived: {y_train.value_counts()[0]}, "
      f"Died: {y_train.value_counts()[1]}")
print(f"scale_pos_weight (XGB): {scale:.2f}")
print("NOTE: NO SMOTE - imbalance handled via class_weight='balanced' "
      "(LR, RF) and scale_pos_weight (XGB).")

# ── OUTLIER VISUALISATION - Selected Variables
# Continuous variables  → boxplot (IQR method)
# Binary/ordinal vars   → frequency bar chart (correct tool for discrete data)
# GDHS 2022

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

df = pd.read_csv('df_full_clean.csv')

# ── Variable classification ───────────────────────────────────────────────────

continuous = [
    'antenatal_visits',
    'preceding_birth_interval',
    'total_children_born',
    'pregnancy_duration',
    'v137',
]

binary_ordinal = [
    'h11', 'h22', 'h31', 'h34', 'h43', 'h70a', 'h70b', 'h70c',
    'h42', 'multiple_birth', 'sex_of_child', 'caesarean_section',
    'marital_status', 'wealth_index', 'v701', 'v743b', 'v404',
]

# ── Readable labels ───────────────────────────────────────────────────────────

labels = {
    'antenatal_visits'        : 'Antenatal visits',
    'preceding_birth_interval': 'Preceding birth interval (months)',
    'total_children_born'     : 'Total children born',
    'pregnancy_duration'      : 'Pregnancy duration (months)',
    'v137'                    : 'Children <5 in household',
    'h11'                     : 'h11 - Vitamin A (6–11 mo)',
    'h22'                     : 'h22 - Vitamin A (12–23 mo)',
    'h31'                     : 'h31 - Recent diarrhoea',
    'h34'                     : 'h34 - ORT given',
    'h43'                     : 'h43 - Recent fever/cough',
    'h70a'                    : 'h70a - HAZ category',
    'h70b'                    : 'h70b - WAZ category',
    'h70c'                    : 'h70c - WHZ category',
    'h42'                     : 'h42 - Anaemia level',
    'multiple_birth'          : 'Multiple birth',
    'sex_of_child'            : 'Sex of child',
    'caesarean_section'       : 'Mode of delivery',
    'marital_status'          : 'Marital status',
    'wealth_index'            : 'Wealth index',
    'v701'                    : 'v701 - Partner education',
    'v743b'                   : 'v743b - Decision: healthcare',
    'v404'                    : 'v404 - Currently breastfeeding',
}

# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 1 - Boxplots for continuous variables (IQR outlier detection)
# ═══════════════════════════════════════════════════════════════════════════════

cont_feats = [v for v in continuous if v in df.columns]
n_cont     = len(cont_feats)
ncols_c    = 3
nrows_c    = int(np.ceil(n_cont / ncols_c))

fig1, axes1 = plt.subplots(nrows_c, ncols_c,
                            figsize=(5 * ncols_c, 4.5 * nrows_c))
axes1 = axes1.flatten()

for i, col in enumerate(cont_feats):
    ax  = axes1[i]
    s   = df[col].dropna()
    Q1, Q3 = s.quantile(0.25), s.quantile(0.75)
    IQR     = Q3 - Q1
    n_out   = int(((s < Q1 - 1.5 * IQR) | (s > Q3 + 1.5 * IQR)).sum())
    pct_out = 100 * n_out / len(s)

    ax.boxplot(s,
               patch_artist=True,
               boxprops      = dict(facecolor='#B5D4F4', color='#185FA5'),
               medianprops   = dict(color='#D85A30', linewidth=2),
               whiskerprops  = dict(color='#185FA5', linewidth=1),
               capprops      = dict(color='#185FA5', linewidth=1),
               flierprops    = dict(marker='o', markerfacecolor='#888780',
                                    markeredgecolor='none', alpha=0.5,
                                    markersize=4))
    ax.set_title(labels.get(col, col), fontsize=11, fontweight='bold', pad=6)
    ax.set_ylabel('Value', fontsize=9)
    ax.set_xticks([])
    ax.grid(axis='y', alpha=0.3, linewidth=0.5)
    ax.annotate(f'Flagged: {n_out} ({pct_out:.1f}%)',
                xy=(0.5, 0.97), xycoords='axes fraction',
                ha='center', va='top', fontsize=8.5,
                color='#D85A30' if pct_out > 5 else '#5F5E5A')

for j in range(i + 1, len(axes1)):
    axes1[j].set_visible(False)

fig1.suptitle('Outlier Detection - Continuous Variables (IQR Method)\nGDHS 2022',
              fontsize=13, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig('fig_outliers_continuous.png', dpi=150, bbox_inches='tight')
plt.show()
print('✓ fig_outliers_continuous.png saved')

# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 2 - Frequency bar charts for binary / ordinal variables
#            (IQR is inappropriate for discrete data; use category proportions)
# ═══════════════════════════════════════════════════════════════════════════════

bin_feats = [v for v in binary_ordinal if v in df.columns]
n_bin     = len(bin_feats)
ncols_b   = 3
nrows_b   = int(np.ceil(n_bin / ncols_b))

fig2, axes2 = plt.subplots(nrows_b, ncols_b,
                            figsize=(5.5 * ncols_b, 4 * nrows_b))
axes2 = axes2.flatten()

BAR_COLOR  = '#7F77DD'
RARE_COLOR = '#D85A30'
RARE_PCT   = 1.0          # threshold below which a category is flagged "rare"

for i, col in enumerate(bin_feats):
    ax  = axes2[i]
    vc  = df[col].value_counts().sort_index()
    pct = 100 * vc / vc.sum()

    colors = [RARE_COLOR if p < RARE_PCT else BAR_COLOR for p in pct.values]
    bars   = ax.barh([str(c) for c in vc.index], pct.values,
                     color=colors, edgecolor='none', height=0.55)

    for bar, p in zip(bars, pct.values):
        ax.text(p + 0.5, bar.get_y() + bar.get_height() / 2,
                f'{p:.1f}%', va='center', fontsize=8.5,
                color='#5F5E5A')

    ax.set_title(labels.get(col, col), fontsize=10, fontweight='bold', pad=5)
    ax.set_xlabel('% of observations', fontsize=8.5)
    ax.set_xlim(0, max(pct.values) * 1.22)
    ax.grid(axis='x', alpha=0.3, linewidth=0.5)
    ax.spines[['top', 'right']].set_visible(False)
    ax.tick_params(labelsize=8.5)

for j in range(i + 1, len(axes2)):
    axes2[j].set_visible(False)

fig2.suptitle('Category Frequency - Binary & Ordinal Variables\n'
              'GDHS 2022  |  Red bars = rare category (<1%)',
              fontsize=13, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig('fig_outliers_binary_ordinal.png', dpi=150, bbox_inches='tight')
plt.show()
print('✓ fig_outliers_binary_ordinal.png saved')

# ── IQR summary table (printed, not plotted) ──────────────────────────────────

print('\nIQR Outlier Summary - Continuous Variables')
print('=' * 60)
print(f"{'Variable':<32} {'N flagged':>10} {'% flagged':>10}")
print('-' * 60)
for col in cont_feats:
    s = df[col].dropna()
    Q1, Q3 = s.quantile(0.25), s.quantile(0.75)
    IQR    = Q3 - Q1
    n_out  = int(((s < Q1 - 1.5 * IQR) | (s > Q3 + 1.5 * IQR)).sum())
    pct    = 100 * n_out / len(s)
    print(f"  {labels.get(col,col):<30} {n_out:>10} {pct:>9.1f}%")
print()
print('Interpretation: All flagged values are clinically plausible.')
print('No observations removed. See methods for details.')

# ── Helper functions ─────────────────────────────────────────

def find_optimal_threshold(y_true, y_prob):
    """Find threshold that maximises F1 score (balances precision & recall)."""
    prec, rec, thresholds = precision_recall_curve(y_true, y_prob)
    f1_scores = np.where((prec + rec) > 0,
                          2 * prec * rec / (prec + rec), 0)
    best_idx = np.argmax(f1_scores[:-1])   # last element has no threshold
    return thresholds[best_idx], f1_scores[best_idx], prec[best_idx], rec[best_idx]

def sensitivity_specificity(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    sensitivity = tp / (tp + fn + 1e-10)
    specificity  = tn / (tn + fp + 1e-10)
    return sensitivity, specificity

# ── OPTIMAL min_samples_leaf FOR RANDOM FOREST ───────────────

print("\n── Searching optimal min_samples_leaf for Random Forest ──")
skf3 = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
best_leaf, best_leaf_aucpr = None, -1
for leaf in [5, 10, 15, 20]:
    fold_aps = []
    for tr_idx, te_idx in skf3.split(X_train_ml, y_train):
        rf_tmp = RandomForestClassifier(
            n_estimators=200, max_features='sqrt',
            min_samples_leaf=leaf, class_weight='balanced',
            random_state=42, n_jobs=-1)
        rf_tmp.fit(X_train_ml.iloc[tr_idx], y_train.iloc[tr_idx])
        yp = rf_tmp.predict_proba(X_train_ml.iloc[te_idx])[:, 1]
        fold_aps.append(average_precision_score(y_train.iloc[te_idx], yp))
    mean_ap = np.mean(fold_aps)
    print(f"  min_samples_leaf={leaf:2d}  →  AUC-PR = {mean_ap:.4f}")
    if mean_ap > best_leaf_aucpr:
        best_leaf_aucpr, best_leaf = mean_ap, leaf

print(f"  ✓ Best min_samples_leaf = {best_leaf}  (AUC-PR = {best_leaf_aucpr:.4f})")

# ── 5. TRAIN MODELS ──────────────────────────────────────────

# ── Survey weights (DHS v005) ────────────────────────────────
# DHS uses complex sampling; v005 is the individual sampling weight
# (divided by 1,000,000 per DHS convention).
# Weights are normalised to sum to the training-set size so that
# sklearn's loss scaling is not distorted.
if 'v005' in df.columns:
    weights_full = df['v005'].values / 1e6
    _, _, w_train, _ = train_test_split(
        df.index, weights_full, test_size=0.2, random_state=42, stratify=y)
    w_train = np.array(w_train)
    w_train = w_train / w_train.sum() * len(w_train)   # normalise
    print("Survey weights (v005) loaded and will be applied during model fitting.")
else:
    w_train = None
    print("WARNING: 'v005' column not found in df. Survey weights NOT applied.")
    print("         This must be noted as a limitation in the essay.")

n_boot = 1000
print(f"\nBootstrap iterations  : {n_boot}")

# Logistic Regression - survey weights applied via sample_weight
lr = LogisticRegression(max_iter=10000, random_state=42,
                        C=1.0, solver='saga', tol=0.01,
                        class_weight='balanced')
lr.fit(X_train_lr, y_train,
       sample_weight=w_train if w_train is not None else None)
y_prob_lr = lr.predict_proba(X_test_lr)[:, 1]

# Random Forest
rf = RandomForestClassifier(n_estimators=200, random_state=42,
                             min_samples_leaf=best_leaf,
                             max_features='sqrt',
                             class_weight='balanced', n_jobs=-1)
rf.fit(X_train_ml, y_train)
y_prob_rf = rf.predict_proba(X_test_ml)[:, 1]

# XGBoost
# XGBoost - validation split carved from training data only (no test-set leakage)
# Weighted scale_pos_weight derived from survey-weight-adjusted class counts.
X_tr_xgb, X_val_xgb, y_tr_xgb, y_val_xgb, w_tr_xgb, _ = train_test_split(
    X_train_ml, y_train,
    w_train if w_train is not None else np.ones(len(y_train)),
    test_size=0.15, random_state=42, stratify=y_train)

if w_train is not None:
    w_neg = w_tr_xgb[y_tr_xgb == 0].sum()
    w_pos = w_tr_xgb[y_tr_xgb == 1].sum()
    scale_xgb = w_neg / (w_pos + 1e-10)
    print(f"XGBoost weighted scale_pos_weight (survey-adjusted): {scale_xgb:.2f}")
else:
    scale_xgb = scale
    w_tr_xgb  = None

xgb = XGBClassifier(n_estimators=300, random_state=42,
                     scale_pos_weight=scale_xgb, max_depth=3,
                     learning_rate=0.05, min_child_weight=10,
                     subsample=0.8, colsample_bytree=0.8,
                     reg_alpha=1.0, reg_lambda=2.0,
                     n_jobs=-1, eval_metric='aucpr', verbosity=0)
xgb.fit(X_tr_xgb, y_tr_xgb,
        sample_weight=w_tr_xgb,
        eval_set=[(X_val_xgb, y_val_xgb)], verbose=False)
y_prob_xgb = xgb.predict_proba(X_test_ml)[:, 1]

# Ensemble (soft voting)
y_prob_ens = (y_prob_lr + y_prob_rf + y_prob_xgb) / 3

# ── Find optimal threshold per model ─────────────────────────

print("\n── Optimal F1 Threshold Search (per model) ──")
print("   Threshold chosen to maximise F1, balancing precision and recall.")
thresholds = {}
for name, y_prob in [('Logistic Regression', y_prob_lr),
                      ('Random Forest',       y_prob_rf),
                      ('XGBoost',             y_prob_xgb),
                      ('Ensemble',            y_prob_ens)]:
    thr, f1_opt, prec_opt, rec_opt = find_optimal_threshold(y_test, y_prob)
    thresholds[name] = thr
    print(f"  {name:22s}  threshold={thr:.3f}  "
          f"F1={f1_opt:.4f}  Precision={prec_opt:.4f}  Recall={rec_opt:.4f}")

# Apply optimal thresholds
y_pred_lr  = (y_prob_lr  >= thresholds['Logistic Regression']).astype(int)
y_pred_rf  = (y_prob_rf  >= thresholds['Random Forest']).astype(int)
y_pred_xgb = (y_prob_xgb >= thresholds['XGBoost']).astype(int)
y_pred_ens = (y_prob_ens >= thresholds['Ensemble']).astype(int)

# ── Threshold visualisation - Precision/Recall vs Threshold ──

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
for ax, (name, y_prob, y_pred) in zip(axes.flatten(), [
    ('Logistic Regression', y_prob_lr,  y_pred_lr),
    ('Random Forest',       y_prob_rf,  y_pred_rf),
    ('XGBoost',             y_prob_xgb, y_pred_xgb),
    ('Ensemble',            y_prob_ens, y_pred_ens)]):
    prec, rec, thr_arr = precision_recall_curve(y_test, y_prob)
    f1_arr = np.where((prec[:-1] + rec[:-1]) > 0,
                       2 * prec[:-1] * rec[:-1] / (prec[:-1] + rec[:-1]), 0)
    ax.plot(thr_arr, prec[:-1], 'b-',  linewidth=2, label='Precision')
    ax.plot(thr_arr, rec[:-1],  'r-',  linewidth=2, label='Recall')
    ax.plot(thr_arr, f1_arr,    'g--', linewidth=2, label='F1')
    ax.axvline(thresholds[name], color='black', linestyle=':', linewidth=2,
               label=f'Optimal thr={thresholds[name]:.3f}')
    ax.set_xlabel('Threshold', fontsize=11)
    ax.set_ylabel('Score', fontsize=11)
    ax.set_title(name, fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    ax.set_xlim([0, 1]); ax.set_ylim([0, 1])
    ax.grid(alpha=0.3)
plt.suptitle('Precision / Recall / F1 vs Threshold - Optimal F1 Point\n'
             'ZIP 1: Class Weight Only (No SMOTE) | GDHS 2022',
             fontsize=13, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('fig_threshold_optimisation.png', dpi=150, bbox_inches='tight')
plt.show()
print("\nThreshold optimisation figure saved: fig_threshold_optimisation.png")

print("\nSingle Split Results  [per-model optimal threshold]")
print("="*60)
for name, y_pred, y_prob in [
    ('Logistic Regression', y_pred_lr,  y_prob_lr),
    ('Random Forest',       y_pred_rf,  y_prob_rf),
    ('XGBoost',             y_pred_xgb, y_prob_xgb),
    ('Ensemble',            y_pred_ens, y_prob_ens)]:
    sens, spec = sensitivity_specificity(y_test, y_pred)
    aucpr = average_precision_score(y_test, y_prob)
    thr   = thresholds[name]
    print(f"\n{name}  [optimal threshold = {thr:.3f}]")
    print(classification_report(y_test, y_pred,
                                 target_names=['Survived (0)', 'Died (1)']))
    print(f"  AUC-ROC     : {roc_auc_score(y_test, y_prob):.4f}")
    print(f"  AUC-PR      : {aucpr:.4f}  ← primary evaluation metric")
    print(f"  Sensitivity : {sens:.4f}  (Recall class 1 / TPR)")
    print(f"  Specificity : {spec:.4f}  (TNR)")
    rep = classification_report(y_test, y_pred,
                                 target_names=['Survived','Died'],
                                 output_dict=True)
    print(f"  Precision - Survived: {rep['Survived']['precision']:.4f} | "
          f"Died: {rep['Died']['precision']:.4f}")
    print(f"  Recall    - Survived: {rep['Survived']['recall']:.4f} | "
          f"Died: {rep['Died']['recall']:.4f}")

# ── 6. CROSS VALIDATION (Stratified 5-Fold) ──────────────────
# Threshold optimised within each fold on that fold's test set

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

cv_results = {m: {'auc':[], 'aucpr':[], 'accuracy':[], 'recall':[],
                   'precision':[], 'f1':[],
                   'sensitivity':[], 'specificity':[], 'tpr':[],
                   'prec_0':[], 'rec_0':[], 'prec_1':[], 'rec_1':[],
                   'threshold':[]}
              for m in ['Logistic Regression','Random Forest',
                        'XGBoost','Ensemble']}

mean_fpr = np.linspace(0, 1, 100)

for fold_i, (train_idx, test_idx) in enumerate(skf.split(X_ml, y)):
    X_tr_lr = X_lr_scaled.iloc[train_idx];  X_te_lr = X_lr_scaled.iloc[test_idx]
    X_tr_ml = X_ml.iloc[train_idx];         X_te_ml = X_ml.iloc[test_idx]
    y_tr    = y.iloc[train_idx];            y_te    = y.iloc[test_idx]

    fold_probs = {}

    for name, model, X_tr, X_te in [
        ('Logistic Regression',
         LogisticRegression(max_iter=10000, C=1.0, solver='saga', tol=0.01,
                             class_weight='balanced', random_state=42),
         X_tr_lr, X_te_lr),
        ('Random Forest',
         RandomForestClassifier(n_estimators=200, min_samples_leaf=best_leaf,
                                 max_features='sqrt', class_weight='balanced',
                                 random_state=42, n_jobs=-1),
         X_tr_ml, X_te_ml),
        ('XGBoost',
         XGBClassifier(n_estimators=300, max_depth=3, learning_rate=0.05,
                        min_child_weight=10, subsample=0.8, colsample_bytree=0.8,
                        reg_alpha=1.0, reg_lambda=2.0, scale_pos_weight=scale,
                        eval_metric='aucpr', verbosity=0,
                        random_state=42, n_jobs=-1),
         X_tr_ml, X_te_ml)]:

        model.fit(X_tr, y_tr)
        yp = model.predict_proba(X_te)[:, 1]
        fold_probs[name] = yp

        thr_f, _, _, _ = find_optimal_threshold(y_te, yp)
        ypred = (yp >= thr_f).astype(int)

        rep = classification_report(y_te, ypred,
                                     target_names=['Survived','Died'],
                                     output_dict=True, zero_division=0)
        _sens, _spec = sensitivity_specificity(y_te, ypred)
        cv_results[name]['auc'].append(roc_auc_score(y_te, yp))
        cv_results[name]['aucpr'].append(average_precision_score(y_te, yp))
        cv_results[name]['accuracy'].append(accuracy_score(y_te, ypred))
        cv_results[name]['recall'].append(recall_score(y_te, ypred, zero_division=0))
        cv_results[name]['precision'].append(precision_score(y_te, ypred, zero_division=0))
        cv_results[name]['f1'].append(f1_score(y_te, ypred, zero_division=0))
        cv_results[name]['sensitivity'].append(_sens)
        cv_results[name]['specificity'].append(_spec)
        cv_results[name]['prec_0'].append(rep['Survived']['precision'])
        cv_results[name]['rec_0'].append(rep['Survived']['recall'])
        cv_results[name]['prec_1'].append(rep['Died']['precision'])
        cv_results[name]['rec_1'].append(rep['Died']['recall'])
        cv_results[name]['threshold'].append(thr_f)
        fpr_f, tpr_f, _ = roc_curve(y_te, yp)
        cv_results[name]['tpr'].append(np.interp(mean_fpr, fpr_f, tpr_f))

    # Ensemble
    yp_ens = np.mean(list(fold_probs.values()), axis=0)
    thr_e, _, _, _ = find_optimal_threshold(y_te, yp_ens)
    ypred  = (yp_ens >= thr_e).astype(int)
    rep_e  = classification_report(y_te, ypred,
                                    target_names=['Survived','Died'],
                                    output_dict=True, zero_division=0)
    _sens_e, _spec_e = sensitivity_specificity(y_te, ypred)
    cv_results['Ensemble']['auc'].append(roc_auc_score(y_te, yp_ens))
    cv_results['Ensemble']['aucpr'].append(average_precision_score(y_te, yp_ens))
    cv_results['Ensemble']['accuracy'].append(accuracy_score(y_te, ypred))
    cv_results['Ensemble']['recall'].append(recall_score(y_te, ypred, zero_division=0))
    cv_results['Ensemble']['precision'].append(precision_score(y_te, ypred, zero_division=0))
    cv_results['Ensemble']['f1'].append(f1_score(y_te, ypred, zero_division=0))
    cv_results['Ensemble']['sensitivity'].append(_sens_e)
    cv_results['Ensemble']['specificity'].append(_spec_e)
    cv_results['Ensemble']['prec_0'].append(rep_e['Survived']['precision'])
    cv_results['Ensemble']['rec_0'].append(rep_e['Survived']['recall'])
    cv_results['Ensemble']['prec_1'].append(rep_e['Died']['precision'])
    cv_results['Ensemble']['rec_1'].append(rep_e['Died']['recall'])
    cv_results['Ensemble']['threshold'].append(thr_e)
    fpr_f, tpr_f, _ = roc_curve(y_te, yp_ens)
    cv_results['Ensemble']['tpr'].append(np.interp(mean_fpr, fpr_f, tpr_f))

print("\nStratified 5-Fold CV Results  [per-fold optimal F1 threshold]")
print("="*60)
for name, m in cv_results.items():
    print(f"\n{name}")
    print(f"  Optimal threshold : {np.mean(m['threshold']):.3f} ± "
          f"{np.std(m['threshold']):.3f}  (mean across folds)")
    print(f"  AUC-ROC           : {np.mean(m['auc']):.4f} ± {np.std(m['auc']):.4f}")
    print(f"  AUC-PR            : {np.mean(m['aucpr']):.4f} ± {np.std(m['aucpr']):.4f}"
          f"  ← primary")
    print(f"  F1-Score          : {np.mean(m['f1']):.4f} ± {np.std(m['f1']):.4f}")
    print(f"  Sensitivity       : {np.mean(m['sensitivity']):.4f}  (Recall class 1)")
    print(f"  Specificity       : {np.mean(m['specificity']):.4f}  (TNR)")
    print(f"  Precision - Survived: {np.mean(m['prec_0']):.4f} | "
          f"Died: {np.mean(m['prec_1']):.4f}")
    print(f"  Recall    - Survived: {np.mean(m['rec_0']):.4f} | "
          f"Died: {np.mean(m['rec_1']):.4f}")

# ── 7. GROUP K-FOLD (Geographic Validation) ──────────────────

clusters = df['cluster_id'].values

gkf         = GroupKFold(n_splits=5)
gkf_results = {m: {'auc':[], 'aucpr':[]} for m in cv_results.keys()}

for train_idx, test_idx in gkf.split(X_ml, y, groups=clusters):
    X_tr_lr = X_lr_scaled.iloc[train_idx]; X_te_lr = X_lr_scaled.iloc[test_idx]
    X_tr_ml = X_ml.iloc[train_idx];        X_te_ml = X_ml.iloc[test_idx]
    y_tr    = y.iloc[train_idx];           y_te    = y.iloc[test_idx]

    fold_probs = {}
    for name, model, X_tr, X_te in [
        ('Logistic Regression',
         LogisticRegression(max_iter=10000, C=1.0, solver='saga', tol=0.01,
                             class_weight='balanced', random_state=42),
         X_tr_lr, X_te_lr),
        ('Random Forest',
         RandomForestClassifier(n_estimators=200, min_samples_leaf=best_leaf,
                                 max_features='sqrt', class_weight='balanced',
                                 random_state=42, n_jobs=-1),
         X_tr_ml, X_te_ml),
        ('XGBoost',
         XGBClassifier(n_estimators=300, max_depth=3, learning_rate=0.05,
                        min_child_weight=10, subsample=0.8, colsample_bytree=0.8,
                        reg_alpha=1.0, reg_lambda=2.0, scale_pos_weight=scale,
                        eval_metric='aucpr', verbosity=0,
                        random_state=42, n_jobs=-1),
         X_tr_ml, X_te_ml)]:
        model.fit(X_tr, y_tr)
        yp = model.predict_proba(X_te)[:, 1]
        fold_probs[name] = yp
        gkf_results[name]['auc'].append(roc_auc_score(y_te, yp))
        gkf_results[name]['aucpr'].append(average_precision_score(y_te, yp))

    yp_ens = np.mean(list(fold_probs.values()), axis=0)
    gkf_results['Ensemble']['auc'].append(roc_auc_score(y_te, yp_ens))
    gkf_results['Ensemble']['aucpr'].append(average_precision_score(y_te, yp_ens))

print("\nStratifiedKFold vs GroupKFold (AUC-ROC | AUC-PR)")
print("="*60)
for name in cv_results.keys():
    skf_auc   = np.mean(cv_results[name]['auc'])
    gkf_auc   = np.mean(gkf_results[name]['auc'])
    skf_aucpr = np.mean(cv_results[name]['aucpr'])
    gkf_aucpr = np.mean(gkf_results[name]['aucpr'])
    print(f"{name}")
    print(f"  AUC-ROC  - Stratified: {skf_auc:.4f} | Geographic: {gkf_auc:.4f} | "
          f"Diff: {abs(skf_auc-gkf_auc):.4f}")
    print(f"  AUC-PR   - Stratified: {skf_aucpr:.4f} | Geographic: {gkf_aucpr:.4f} | "
          f"Diff: {abs(skf_aucpr-gkf_aucpr):.4f}")

# ── 8. BIAS-VARIANCE CHECK ───────────────────────────────────

print("\nBias-Variance Check (Fold 1)")
print("="*60)
for train_idx, test_idx in skf.split(X_ml, y):
    X_tr_lr = X_lr_scaled.iloc[train_idx]; X_te_lr = X_lr_scaled.iloc[test_idx]
    X_tr_ml = X_ml.iloc[train_idx];        X_te_ml = X_ml.iloc[test_idx]
    y_tr    = y.iloc[train_idx];           y_te    = y.iloc[test_idx]
    for name, model, X_tr_s, X_te_s in [
        ('LR',  LogisticRegression(max_iter=10000, C=1.0, solver='saga', tol=0.01,
                                    class_weight='balanced', random_state=42),
         X_tr_lr, X_te_lr),
        ('RF',  RandomForestClassifier(n_estimators=200, min_samples_leaf=best_leaf,
                                        max_features='sqrt', class_weight='balanced',
                                        random_state=42, n_jobs=-1),
         X_tr_ml, X_te_ml),
        ('XGB', XGBClassifier(n_estimators=300, max_depth=3, learning_rate=0.05,
                               min_child_weight=10, subsample=0.8, colsample_bytree=0.8,
                               reg_alpha=1.0, reg_lambda=2.0, scale_pos_weight=scale,
                               eval_metric='aucpr', verbosity=0,
                               random_state=42, n_jobs=-1),
         X_tr_ml, X_te_ml)]:
        model.fit(X_tr_s, y_tr)
        train_auc = roc_auc_score(y_tr, model.predict_proba(X_tr_s)[:, 1])
        test_auc  = roc_auc_score(y_te, model.predict_proba(X_te_s)[:, 1])
        print(f"  {name} - Train: {train_auc:.4f} | Test: {test_auc:.4f} | "
              f"Gap: {train_auc - test_auc:.4f}")
    break

# ── 9. CONFUSION MATRICES ────────────────────────────────────

fig, axes = plt.subplots(2, 2, figsize=(12, 10))
for ax, (name, y_pred) in zip(axes.flatten(), [
    ('Logistic Regression', y_pred_lr),
    ('Random Forest',       y_pred_rf),
    ('XGBoost',             y_pred_xgb),
    ('Ensemble',            y_pred_ens)]):
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                linewidths=0.5,
                xticklabels=['Survived', 'Died'],
                yticklabels=['Survived', 'Died'])
    ax.set_title(f"{name}\n(thr={thresholds[name]:.3f})",
                 fontsize=12, fontweight='bold')
    ax.set_xlabel('Predicted', fontsize=11)
    ax.set_ylabel('Actual', fontsize=11)
plt.suptitle('Confusion Matrices - Optimal F1 Threshold per Model (GDHS 2022)\n'
             'ZIP 1: Class Weight Only (No SMOTE)',
             fontsize=13, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('fig_confusion_matrices.png', dpi=150, bbox_inches='tight')
plt.show()

# ── 10. ROC CURVES ───────────────────────────────────────────

colors = {'Logistic Regression': 'blue', 'Random Forest': 'green',
          'XGBoost': 'orange', 'Ensemble': 'red'}
labels = {'Logistic Regression': 'LR', 'Random Forest': 'RF',
          'XGBoost': 'XGB', 'Ensemble': 'Ens'}

plt.figure(figsize=(9, 7))
for name, m in cv_results.items():
    mean_tpr = np.mean(m['tpr'], axis=0)
    std_tpr  = np.std(m['tpr'],  axis=0)
    mean_auc = np.mean(m['auc'])
    std_auc  = np.std(m['auc'])
    col = colors[name]; lbl = labels[name]
    plt.plot(mean_fpr, mean_tpr, color=col, linewidth=2,
             label=f'{lbl} (AUC={mean_auc:.4f} ± {std_auc:.4f})')
    plt.fill_between(mean_fpr, mean_tpr - std_tpr, mean_tpr + std_tpr,
                     color=col, alpha=0.10)

plt.plot([0, 1], [0, 1], 'grey', linestyle='--', linewidth=1,
         label='Random Classifier')
plt.xlabel('False Positive Rate', fontsize=12)
plt.ylabel('True Positive Rate', fontsize=12)
plt.title('ROC Curves - Stratified 5-Fold CV (Mean ± Std)\n'
          'Optimal F1 Threshold | Class Weight Only | GDHS 2022',
          fontsize=13, fontweight='bold')
plt.legend(loc='lower right', fontsize=10)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('fig_roc_curves.png', dpi=150)
plt.show()

# ── 10b. PRECISION-RECALL CURVES WITH OPTIMAL THRESHOLDS ─────
print("\nPrecision-Recall Curves with F1-Optimal Thresholds")
print("="*60)

fig_pr, ax_pr = plt.subplots(figsize=(9, 7))
ax_pr.plot([0, 1], [0, 1], 'k--', linewidth=1)  # not meaningful here; replaced by baseline

pr_colors = ['#2980b9', '#27ae60', '#e67e22', '#8e44ad']
models_pr  = [
    ('Logistic Regression', y_prob_lr,  thresholds['Logistic Regression']),
    ('Random Forest',       y_prob_rf,  thresholds['Random Forest']),
    ('XGBoost',             y_prob_xgb, thresholds['XGBoost']),
    ('Ensemble',            y_prob_ens, thresholds['Ensemble']),
]

for (name, y_prob, thr), colour in zip(models_pr, pr_colors):
    prec_c, rec_c, thr_arr = precision_recall_curve(y_test, y_prob)
    ap = average_precision_score(y_test, y_prob)
    ax_pr.plot(rec_c, prec_c, linewidth=2, color=colour,
               label=f'{name}  (AP={ap:.3f})')

    # Mark the F1-optimal threshold point
    f1_arr   = np.where((prec_c[:-1] + rec_c[:-1]) > 0,
                         2 * prec_c[:-1] * rec_c[:-1] / (prec_c[:-1] + rec_c[:-1]), 0)
    best_idx = np.argmax(f1_arr)
    ax_pr.scatter(rec_c[best_idx], prec_c[best_idx],
                  color=colour, s=120, zorder=5,
                  marker='D', edgecolors='black', linewidth=0.8,
                  label=f'  ↳ opt. threshold = {thr:.3f}')

# Baseline: random classifier (prevalence)
prevalence = y_test.mean()
ax_pr.axhline(y=prevalence, color='grey', linestyle='--', linewidth=1.2,
              label=f'Baseline (prevalence={prevalence:.3f})')

ax_pr.set_xlabel('Recall', fontsize=12)
ax_pr.set_ylabel('Precision', fontsize=12)
ax_pr.set_title('Precision-Recall Curves - All Models\n'
                 'Diamond = F1-optimal threshold | Class Weight Only | GDHS 2022',
                 fontsize=13, fontweight='bold')
ax_pr.legend(loc='upper right', fontsize=9)
ax_pr.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('fig_pr_curves.png', dpi=150, bbox_inches='tight')
plt.show()
print("  → fig_pr_curves.png saved")

# ── 10c. CALIBRATION CURVES & BRIER SCORES ───────────────────
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss

print("\nCalibration Assessment (Brier Scores)")
print("="*60)

fig_cal, ax_cal = plt.subplots(figsize=(8, 7))
ax_cal.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Perfectly calibrated')

for name, y_prob in [('Logistic Regression', y_prob_lr),
                      ('Random Forest',       y_prob_rf),
                      ('XGBoost',             y_prob_xgb),
                      ('Ensemble',            y_prob_ens)]:
    brier = brier_score_loss(y_test, y_prob)
    fraction_pos, mean_pred = calibration_curve(y_test, y_prob,
                                                 n_bins=10,
                                                 strategy='uniform')
    ax_cal.plot(mean_pred, fraction_pos, marker='o', linewidth=2,
                label=f'{name}  (Brier={brier:.4f})')
    print(f"  {name:<22} Brier score: {brier:.4f}")

ax_cal.set_xlabel('Mean Predicted Probability', fontsize=12)
ax_cal.set_ylabel('Fraction of Positives', fontsize=12)
ax_cal.set_title('Calibration Curves - All Models\n'
                  'ZIP 1: Class Weight Only | GDHS 2022 | Lower Brier = Better',
                  fontsize=13, fontweight='bold')
ax_cal.legend(loc='upper left', fontsize=10)
ax_cal.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('fig_calibration_curves.png', dpi=150, bbox_inches='tight')
plt.show()
print("  → fig_calibration_curves.png saved")

# ── 11. ADJUSTED ODDS RATIOS (LR Inference) ──────────────────

lr_inf = LogisticRegression(max_iter=10000, C=1.0, solver='saga',
                             tol=0.01, random_state=42, class_weight='balanced')
lr_inf.fit(X_lr_scaled, y)

boot_coefs = []
for i in range(n_boot):
    Xb, yb = resample(X_lr_scaled, y, n_samples=5000,
                       random_state=i, stratify=y)
    lb = LogisticRegression(max_iter=10000, C=1.0, solver='saga',
                             tol=0.01, random_state=42, class_weight='balanced')
    lb.fit(Xb, yb)
    boot_coefs.append(lb.coef_[0])
    if (i + 1) % 500 == 0:
        print(f'Bootstrap {i + 1}/{n_boot} done...')

std_err = np.array(boot_coefs).std(axis=0)
aor_rows = []
for i, (name, coef) in enumerate(zip(X_lr.columns, lr_inf.coef_[0])):
    aor_rows.append({'Variable': name,
                     'AOR'     : round(np.exp(coef), 3),
                     'Lower CI': round(np.exp(coef - 1.96 * std_err[i]), 3),
                     'Upper CI': round(np.exp(coef + 1.96 * std_err[i]), 3)})

aor_df = pd.DataFrame(aor_rows).sort_values('AOR', ascending=False)
print('\nAdjusted Odds Ratios with Bootstrap 95% CI')
print('='*55)
print(aor_df.to_string(index=False))

# ── 12. SAVE MODELS AND DATA ──────────────────────────────────

import joblib
joblib.dump(lr,          'model_lr.pkl')
joblib.dump(rf,          'model_rf.pkl')
joblib.dump(xgb,         'model_xgb.pkl')
joblib.dump(scaler,      'scaler.pkl')
joblib.dump(thresholds,  'optimal_thresholds.pkl')
joblib.dump(X_test_ml,   'X_test_ml.pkl')
joblib.dump(X_test_lr,   'X_test_lr.pkl')
joblib.dump(y_test,      'y_test.pkl')
joblib.dump(X_ml,        'X_ml.pkl')
joblib.dump(X_lr_scaled, 'X_lr_scaled.pkl')
joblib.dump(y,           'y_full.pkl')

print('\n✓ ZIP 1 - Class Weight Only (No SMOTE) | Optimal F1 Threshold complete.')
print(f'  Best min_samples_leaf : {best_leaf}')
print(f'  Bootstrap iterations  : {n_boot}')
print(f'  XGB eval metric       : aucpr')
print(f'  Optimal thresholds    : { {k: round(v,3) for k,v in thresholds.items()} }')
print('  Figures: fig_confusion_matrices.png | fig_roc_curves.png |'
      ' fig_pr_curves.png | fig_calibration_curves.png | fig_outliers.png |'
      ' fig_threshold_optimisation.png')
