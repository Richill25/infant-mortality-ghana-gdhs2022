# ============================================================
# 01_EDA AND FEATURE SELECTION - FINAL
# Infant Mortality in Ghana - GDHS 2022
#
# Analytical choices documented here:
#
# SAMPLE RESTRICTION
#   • Restricted to bidx=1 (most recent birth per woman)
#     n = 10,356 births, 237 infant deaths, IMR = 22.9/1,000
#   • The full birth-history IMR is 39.1/1,000 - the bidx=1
#     figure is a selection artefact (younger, lower-risk
#     mothers over-represented) and must NOT be interpreted
#     as the national IMR estimate.
#   • A transparency visual is produced comparing both IMRs.
#
# MATERNITY-CARE VARIABLES (m14, m15, m17)
#   • Only collected for births in the 3-year recall window
#     (2019–2022 in GDHS 2022); older bidx=1 births have
#     NaN by survey design - this is structurally non-random.
#   • Missingness is coded as an informative category
#     ("not_in_recall_window") rather than imputed.
#
# PROTECTED FEATURES (13 variables)
#   • Theoretically grounded; always included regardless of
#     statistical screening.
#   • Excluded from protection: v218 (r=0.94 with v201),
#     v013 (derived from v012), v025 (non-significant),
#     bidx/birth_order (constant=1 after bidx=1 restriction,
#     zero variance, undefined correlation - carries no information).
#
# LEAKAGE VARIABLES
#   • 31 variables flagged and fully excluded - see Section 4.
#
# CANDIDATE POOL
#   • All variables with <40% missingness after bidx=1
#     restriction, minus admin/ID/weight/leakage/protected.
#   • Near-constant (>97%) and derived/redundant columns
#     further removed.
#
# IMPUTATION
#   • Numeric with moderate missingness: KNN (k=5)
#   • Categorical: mode-fill
#   • Maternity variables: informative 'not_in_recall_window'
#   • preceding_birth_interval: 0 (first birth semantics)
#
# FEATURE SELECTION PIPELINE
#   • Stage 1 - Mutual Information (MI): non-parametric,
#     captures non-linear associations
#   • Stage 2 - PCA: dimensionality structure
#   • Stage 3 - BIC forward stepwise logistic regression
#   • Consensus: variable enters final list if it appears
#     in ≥2 of {MI>0, p<0.05 univariate, BIC-selected}
#     Protected variables always included.
# ============================================================

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chi2_contingency, mannwhitneyu
from scipy import stats as sp_stats
from sklearn.impute import KNNImputer
from sklearn.feature_selection import mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from matplotlib.patches import Patch
import warnings
warnings.filterwarnings('ignore')

# ── Plot style ───────────────────────────────────────────────
plt.rcParams.update({
    'figure.dpi'        : 150,
    'axes.spines.top'   : False,
    'axes.spines.right' : False,
    'font.family'       : 'DejaVu Sans',
    'axes.titlepad'     : 10,
})
C_SURV  = '#2ecc71'
C_DIED  = '#e74c3c'
C_PROT  = '#8e44ad'
C_CAND  = '#2980b9'
C_WARN  = '#e67e22'
C_GREY  = '#bdc3c7'

SEP = "=" * 65
print(SEP)
print("  INFANT MORTALITY IN GHANA - GDHS 2022")
print("  Notebook 01: EDA & Feature Selection")
print(SEP)


# ════════════════════════════════════════════════════════════
# 1.  LOAD RAW DATA
# ════════════════════════════════════════════════════════════
print("\n[1] Loading raw GDHS 2022 data …")
df_raw = pd.read_stata('GHBR8CFL.DTA', convert_categoricals=True)
df_num = pd.read_stata('GHBR8CFL.DTA', convert_categoricals=False)
print(f"    Raw shape: {df_raw.shape[0]:,} rows × {df_raw.shape[1]:,} columns")


# ════════════════════════════════════════════════════════════
# 2.  CONSTRUCT OUTCOME VARIABLE (FULL DATASET FIRST)
# ════════════════════════════════════════════════════════════
print("\n[2] Constructing outcome variable …")

b5_alive  = (df_raw['b5'].astype(str).str.lower() == 'yes')
b7_months = pd.to_numeric(df_num['b7'], errors='coerce')
df_raw['infant_death'] = (~b5_alive & (b7_months <= 11)).astype(int)

imr_full = df_raw['infant_death'].mean() * 1000
print(f"    Full birth history IMR : {imr_full:.1f} per 1,000 live births")
print(f"    Full birth history n   : {len(df_raw):,}")


# ════════════════════════════════════════════════════════════
# 3.  RESTRICT TO MOST RECENT BIRTH (bidx = 1)
# ════════════════════════════════════════════════════════════
print("\n[3] Restricting to bidx=1 (most recent birth per woman) …")
df = df_raw[df_raw['bidx'] == 1].copy()
df_n = df_num[df_num['bidx'] == 1].copy()   # numeric-coded twin

# Re-derive outcome on restricted set
b5_alive_r  = (df['b5'].astype(str).str.lower() == 'yes')
b7_months_r = pd.to_numeric(df_n['b7'], errors='coerce')
df['infant_death'] = (~b5_alive_r & (b7_months_r <= 11)).astype(int)

imr_r1  = df['infant_death'].mean() * 1000
n_r1    = len(df)
deaths_r1 = df['infant_death'].sum()
print(f"    bidx=1 sample          : {n_r1:,}")
print(f"    Infant deaths (bidx=1) : {deaths_r1}")
print(f"    IMR (bidx=1)           : {imr_r1:.1f} per 1,000 live births")
print()
print("    ⚠  NOTE: IMR 22.9 < SDG target of 25.")
print("       This is a SELECTION ARTEFACT - younger/lower-risk")
print("       mothers are over-represented in the most recent birth.")
print("       The population IMR (full history) is 39.1/1,000.")
print("       This restriction is made solely to obtain complete")
print("       maternal-care data (m14, m15, m17).")


# ════════════════════════════════════════════════════════════
# 4.  LEAKAGE VARIABLE AUDIT
# ════════════════════════════════════════════════════════════
print("\n[4] Documenting leakage variables …")
LEAKAGE = {
    'b5' : 'child_alive - IS the outcome base',
    'b6' : 'age_at_death - encodes death directly',
    'b7' : 'age_at_death_months - direct outcome',
    'b13': 'flag_age_at_death - meta of b7',
    'b8' : 'current_age_child - observed only if alive',
    'b9' : 'child_lives_with_whom - post-outcome',
    'b16': 'child_line_number_HH - alive-only field',
    'b19': 'age_in_months_since_birth - post-outcome',
    'b12': 'succeeding_birth_interval - requires future birth',
    'b15': 'live_birth_between_births - post-outcome',
    'v206': 'sons_who_died - cumulative death count',
    'v207': 'daughters_who_died - cumulative death count',
    'v202': 'sons_at_home - post-birth living status',
    'v203': 'daughters_at_home - post-birth living status',
    'v204': 'sons_elsewhere - post-birth',
    'v205': 'daughters_elsewhere - post-birth',
    'v208': 'births_in_last_5yrs - includes index birth',
    'v209': 'births_in_last_year',
    'v210': 'births_in_month_of_interview',
    'v219': 'living_children_plus_preg - derived from b5',
    'v220': 'living_children_plus_preg_grouped',
    'v222': 'last_birth_to_interview_months - post-outcome',
    'v237': 'birth_between_last_and_interview',
    'v238': 'births_in_last_3yrs',
    'v218': 'living_children - r=0.94 with total_children_born',
    'm80' : 'pregnancy_outcome - directly labels outcome',
    'b10' : 'completeness_info_flag',
    'b17' : 'day_of_birth - admin',
    'b18' : 'CDC_of_birth - admin',
    'b2'  : 'year_of_birth - time ID',
    'b3'  : 'date_of_birth_CMC - time ID',
    'bord': 'birth_order_duplicate_of_bidx',
    'hw13': 'result_of_height_weight_measurement - value=1 means Dead, encodes outcome directly',
}
print(f"    {len(LEAKAGE)} leakage/problematic variables documented & excluded")


# ════════════════════════════════════════════════════════════
# 5.  DEFINE PROTECTED FEATURES
# ════════════════════════════════════════════════════════════
print("\n[5] Defining protected (theoretically grounded) features …")
# Excluded from protection:
#   v218 living_children - r=0.94 with total_children_born
#   v013 mothers_age_group - derived from v012
#   v025 residence_type - non-significant in original analysis
PROTECTED = {
    # mothers_age (v012), education_level (v106), and place_of_delivery (m15)
    # removed: MI = 0.000 in this bidx=1 restricted dataset - they carry no
    # measurable information about the outcome and would only consume EPV budget.
    # All remaining 10 are retained as theoretically grounded predictors.
    'b0'  : 'multiple_birth',
    'b4'  : 'sex_of_child',
    'b11' : 'preceding_birth_interval',
    'b20' : 'pregnancy_duration',
    'm14' : 'antenatal_visits',
    'm17' : 'caesarean_section',
    'v024': 'region',
    'v190': 'wealth_index',
    'v201': 'total_children_born',
    'v501': 'marital_status',
}
GKF = {'v001': 'cluster_id', 'v023': 'strata_id'}
PROT_NAMES = list(PROTECTED.values())
print(f"    Protected predictors : {len(PROTECTED)}")
print(f"    GKF identifiers      : cluster_id, strata_id (NOT used as predictors)")


# ════════════════════════════════════════════════════════════
# 6.  BUILD CANDIDATE VARIABLE POOL
# ════════════════════════════════════════════════════════════
print("\n[6] Building candidate variable pool …")
ADMIN_EXCL = {
    'caseid','bidxp','midx','midxp','hidx','hidxa',
    'v000','v001','v002','v003','v004','v005',
    'v006','v007','v008','v008a','v009','v010','v011',
    'v014','v015','v016','v017','v018','v019','v019a',
    'v020','v021','v022','v026','v027','v028',
    'v029','v030','v031','v032','v034',
    'awfactt','awfactu','awfactr','awfacte','awfactw',
    'v191','v190a','v191a',
    'v139','v140','v141',
    'v400','v417','v418','v418a','v419','v420','v421',
    'v101','v102','v107','v133','v149',
    'h25d','h25m','h25y','v447a','v452a','v452b','v452c',
    'b1','b21','v013','v025',
    'bidx',       # used for restriction; also removed from PROTECTED (constant=1)
    'bidxp',      # pregnancy index
    'v224','v246','v235','v211',  # history meta / time IDs
}

miss = df.isnull().mean()
MISS_THRESH = 0.40

cands_raw = [
    c for c in df.columns
    if miss[c] < MISS_THRESH
    and c not in ADMIN_EXCL
    and c not in LEAKAGE
    and c not in PROTECTED
    and c not in GKF
    and c != 'infant_death'
]
print(f"    Columns in raw data         : {df_raw.shape[1]:,}")
print(f"    Columns <40% missing (b1)   : {(miss < MISS_THRESH).sum()}")
print(f"    Candidates after exclusions : {len(cands_raw)}")


# ════════════════════════════════════════════════════════════
# 7.  ASSEMBLE WORKING DATAFRAME
# ════════════════════════════════════════════════════════════
print("\n[7] Assembling working dataframe …")
all_cols = (
    list(PROTECTED.keys())
    + cands_raw
    + list(GKF.keys())
    + ['infant_death']
)
all_cols = [c for c in all_cols if c in df.columns]
df = df[all_cols].copy()
df_n = df_n[[c for c in all_cols if c in df_n.columns]].copy()

# Rename protected + GKF to human-readable
df   = df.rename(columns={**PROTECTED, **GKF})
df_n = df_n.rename(columns={**PROTECTED, **GKF})
print(f"    Working dataframe: {df.shape}")


# ════════════════════════════════════════════════════════════
# 8.  TRANSPARENCY VISUAL - IMR SELECTION BIAS
# ════════════════════════════════════════════════════════════
print("\n[8] Transparency visual: IMR comparison …")

# IMR by region - full vs bidx=1
# Cast v024 to str to avoid Categorical groupby issues with Stata import
imr_full_reg = (
    df_raw.assign(v024=df_raw['v024'].astype(str))
    .groupby('v024')['infant_death']
    .agg(deaths='sum', births='count')
    .assign(IMR=lambda x: x['deaths']/x['births']*1000)
    .rename_axis('region').reset_index()
    .sort_values('IMR', ascending=False)
)
imr_r1_reg = (
    df.groupby('region')['infant_death']
    .agg(deaths='sum', births='count')
    .assign(IMR=lambda x: x['deaths']/x['births']*1000)
    .reset_index()
)
imr_merged = imr_full_reg.merge(imr_r1_reg, on='region',
                                 suffixes=('_full','_r1'))
imr_merged = imr_merged.sort_values('IMR_full', ascending=True)

fig, axes = plt.subplots(1, 2, figsize=(15, 7))

# Panel A: national IMR comparison
ax = axes[0]
bars_data = [
    ('Full birth history\n(n=34,663)', imr_full, C_DIED),
    ('Most-recent birth only\n(bidx=1, n=10,356)', imr_r1, C_WARN),
    ('SDG 2030 Target', 25, C_CAND),
]
bars = ax.bar([b[0] for b in bars_data],
              [b[1] for b in bars_data],
              color=[b[2] for b in bars_data],
              edgecolor='white', linewidth=1.5, width=0.55)
ax.bar_label(bars, fmt='%.1f', fontsize=12, padding=4, fontweight='bold')
ax.axhline(25, color=C_CAND, ls='--', lw=1.5, alpha=0.6)
ax.set_ylabel('IMR per 1,000 live births', fontsize=12)
ax.set_title('National IMR - Full History vs Restricted Sample\n'
             'The 22.9 is a selection artefact, NOT national progress',
             fontsize=12, fontweight='bold')
ax.set_ylim(0, 50)

# Panel B: regional comparison
x  = np.arange(len(imr_merged))
w  = 0.38
ax2 = axes[1]
ax2.barh(x + w/2, imr_merged['IMR_full'], w, color=C_DIED, label='Full history', alpha=0.85)
ax2.barh(x - w/2, imr_merged['IMR_r1'],   w, color=C_WARN, label='bidx=1 only', alpha=0.85)
ax2.set_yticks(x)
ax2.set_yticklabels(imr_merged['region'], fontsize=9)
ax2.axvline(25, color=C_CAND, ls='--', lw=1.5, label='SDG target')
ax2.set_xlabel('IMR per 1,000 births', fontsize=12)
ax2.set_title('Regional IMR: Full History vs bidx=1\n(Restriction under-estimates risk in all regions)',
              fontsize=12, fontweight='bold')
ax2.legend(fontsize=10)

plt.suptitle('Sample Restriction Transparency - GDHS 2022',
             fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('fig_00_imr_selection_bias.png', dpi=150, bbox_inches='tight')
plt.show()
print("    → fig_00_imr_selection_bias.png")


# ════════════════════════════════════════════════════════════
# 9.  MISSINGNESS OVERVIEW
# ════════════════════════════════════════════════════════════
print("\n[9] Missingness overview …")

miss_raw = df_raw.isnull().mean() * 100
miss_work = df.isnull().mean() * 100

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Panel A: full dataset missingness bands
bands = [
    (miss_raw > 80).sum(),
    ((miss_raw > 50) & (miss_raw <= 80)).sum(),
    ((miss_raw > 40) & (miss_raw <= 50)).sum(),
    ((miss_raw > 10) & (miss_raw <= 40)).sum(),
    (miss_raw <= 10).sum(),
]
band_labels = ['>80%','50–80%','40–50%','10–40%','≤10%']
band_cols   = ['#c0392b','#e67e22','#f39c12','#27ae60','#2980b9']
b0 = axes[0].bar(band_labels, bands, color=band_cols, edgecolor='white', lw=1.5)
axes[0].bar_label(b0, fmt='%d', fontsize=11, padding=3)
axes[0].set_xlabel('Missingness Band', fontsize=12)
axes[0].set_ylabel('Number of Variables', fontsize=12)
axes[0].set_title('Variable Missingness - Full Raw Dataset\n(1,251 columns, GDHS 2022)',
                   fontsize=13, fontweight='bold')

# Panel B: working dataset after restriction
miss_with = miss_work[miss_work > 0].sort_values(ascending=False).head(30)
colors_w = [C_PROT if v in PROT_NAMES else C_CAND for v in miss_with.index]
bw = axes[1].barh(miss_with.index, miss_with.values,
                   color=colors_w, edgecolor='white')
axes[1].axvline(40, color='red', ls='--', lw=1.5, label='40% threshold')
axes[1].set_xlabel('Missing Values (%)', fontsize=12)
axes[1].set_title('Pre-Imputation Missingness in Working Dataset\n(bidx=1 restricted, top 30 shown)',
                   fontsize=13, fontweight='bold')
axes[1].legend(handles=[
    Patch(color=C_PROT, label='Protected feature'),
    Patch(color=C_CAND, label='Candidate variable'),
    plt.Line2D([0],[0], color='red', ls='--', label='40% threshold'),
], fontsize=9)
plt.tight_layout()
plt.savefig('fig_01_missingness_overview.png', dpi=150, bbox_inches='tight')
plt.show()
print("    → fig_01_missingness_overview.png")


# ════════════════════════════════════════════════════════════
# 10. MISSING VALUE TREATMENT
# ════════════════════════════════════════════════════════════
print("\n[10] Imputation & missing value treatment …")

# ── preceding_birth_interval: missing = first birth ──────────
# Cast to float first - Stata imports this as Categorical which
# rejects new fill values directly.
df['preceding_birth_interval'] = (
    pd.to_numeric(df['preceding_birth_interval'], errors='coerce').fillna(0)
)
print("    preceding_birth_interval → 0 (first-birth semantics)")

# ── Maternity variables: informative missingness category ────
# Missing = birth not in 3-year recall window (2019–2022)
# This is structurally non-random → do NOT impute.
# Convert via object dtype first to avoid Categorical setitem error.
RECALL_CAT = 'not_in_recall_window'
df['antenatal_visits']  = df['antenatal_visits'].astype(object).where(
    df['antenatal_visits'].notna(), RECALL_CAT).astype(str)
df['caesarean_section'] = df['caesarean_section'].astype(object).where(
    df['caesarean_section'].notna(), RECALL_CAT).astype(str)

# Recode antenatal_visits numeric values properly
# (will be treated as ordinal category - not imputed as numeric)
# Note: place_of_delivery removed from PROTECTED (MI=0) so not treated here.
print(f"    antenatal_visits / caesarean_section")
print(f"    → missing coded as '{RECALL_CAT}' (informative, not imputed)")

# ── pregnancy_duration: ensure numeric ───────────────────────
df['pregnancy_duration'] = pd.to_numeric(
    df_n['pregnancy_duration'] if 'pregnancy_duration' in df_n.columns
    else df['pregnancy_duration'], errors='coerce')

# ── Identify remaining candidate columns ─────────────────────
fixed = PROT_NAMES + ['cluster_id','strata_id','infant_death']
cand_cols = [c for c in df.columns if c not in fixed]

# ── Drop near-constant columns (>97% same value) ─────────────
low_var = []
for c in cand_cols:
    vc = df[c].value_counts(normalize=True, dropna=True)
    if len(vc) > 0 and vc.iloc[0] > 0.97:
        low_var.append(c)
df = df.drop(columns=low_var, errors='ignore')
cand_cols = [c for c in cand_cols if c not in low_var]
print(f"    Dropped {len(low_var)} near-constant columns (>97% same value)")

# ── Drop derived / redundant within candidates ───────────────
DERIVED_DROP = [
    'v502','v513',                      # derived from v501, v512
    'v625','v625a','v626','v626a',      # exposure/unmet-need alt definitions
    'v531','v532',                      # imputed age-first-sex + flag
    'v527','v528','v529','v530','v536', # time-since-last-sex multi-encodings
    'v614',                             # ideal children grouped = v613
    'v503',                             # num unions - collinear v501
    'v023',                             # already captured in strata_id
]
df = df.drop(columns=DERIVED_DROP, errors='ignore')
cand_cols = [c for c in cand_cols if c not in DERIVED_DROP]
print(f"    Dropped {len(DERIVED_DROP)} derived/redundant columns")

# ── Separate numeric vs categorical ──────────────────────────
cat_cands, num_cands = [], []
for c in cand_cols:
    if c not in df.columns: continue
    if df[c].dtype in ['object','category'] or df[c].nunique(dropna=True) <= 12:
        cat_cands.append(c)
    else:
        num_cands.append(c)
print(f"    Numeric candidates   : {len(num_cands)}")
print(f"    Categorical candidates: {len(cat_cands)}")

# ── KNN impute numeric candidates with missingness ───────────
num_miss = [c for c in num_cands if df[c].isnull().any()]
if num_miss:
    print(f"    KNN-imputing {len(num_miss)} numeric columns (k=5) …")
    knn = KNNImputer(n_neighbors=5)
    df[num_miss] = knn.fit_transform(df[num_miss])
    print("    KNN imputation complete")

# ── Mode-fill categorical candidates ─────────────────────────
cat_miss = [c for c in cat_cands if c in df.columns and df[c].isnull().any()]
for c in cat_miss:
    mode_val = df[c].mode(dropna=True)
    if len(mode_val) > 0:
        df[c] = df[c].fillna(mode_val.iloc[0])
print(f"    Mode-filled {len(cat_miss)} categorical columns")
print(f"    Remaining NaN after imputation: {df.isnull().sum().sum()}")


# ════════════════════════════════════════════════════════════
# 11. LABEL-ENCODE FOR ANALYSIS
# ════════════════════════════════════════════════════════════
print("\n[11] Label-encoding all variables for statistical analysis …")
df_enc = df.copy()
le = LabelEncoder()
obj_cols = df_enc.select_dtypes(include=['object','category']).columns.tolist()
for c in obj_cols:
    df_enc[c] = le.fit_transform(df_enc[c].astype(str))

# Force all to numeric
for c in df_enc.columns:
    df_enc[c] = pd.to_numeric(df_enc[c], errors='coerce')
df_enc = df_enc.fillna(df_enc.median())

ALL_PREDS  = [c for c in df_enc.columns
              if c not in ('infant_death','cluster_id','strata_id')]
died_e     = df_enc[df_enc['infant_death'] == 1]
survived_e = df_enc[df_enc['infant_death'] == 0]

print(f"    Total predictors available: {len(ALL_PREDS)}")


# ════════════════════════════════════════════════════════════
# 12. OUTCOME & CLASS DISTRIBUTION
# ════════════════════════════════════════════════════════════
print("\n[12] Outcome and class distribution …")

fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# Panel A: class pie
counts = df['infant_death'].value_counts().sort_index()
axes[0].pie(counts, labels=['Survived (0)','Died (1)'],
            colors=[C_SURV, C_DIED], explode=(0, 0.07),
            autopct='%1.2f%%', startangle=90,
            textprops={'fontsize':13},
            wedgeprops={'linewidth':1.5,'edgecolor':'white'})
axes[0].set_title(f'Class Distribution\n(n={n_r1:,}, bidx=1)',
                   fontsize=13, fontweight='bold')

# Panel B: IMR by region (bidx=1)
imr_reg = (
    df.groupby('region')['infant_death']
    .agg(deaths='sum', births='count')
    .assign(IMR=lambda x: round(x['deaths']/x['births']*1000, 1))
    .reset_index().sort_values('IMR')
)
bar_c = [C_DIED if x > 25 else C_SURV for x in imr_reg['IMR']]
bars_r = axes[1].barh(imr_reg['region'], imr_reg['IMR'],
                       color=bar_c, edgecolor='white')
axes[1].axvline(25, color='navy', ls='--', lw=1.5, label='SDG Target (25)')
axes[1].bar_label(bars_r, fmt='%.1f', fontsize=8, padding=2)
axes[1].set_xlabel('IMR per 1,000 births', fontsize=11)
axes[1].set_title('IMR by Region (bidx=1)\n⚠ Restricted sample - under-estimates true IMR',
                   fontsize=12, fontweight='bold')
axes[1].legend(fontsize=10)

# Panel C: birth year distribution
b2_vals = df_num[df_num['bidx']==1]['b2'].dropna()
axes[2].hist(b2_vals, bins=range(int(b2_vals.min()), int(b2_vals.max())+2),
             color=C_CAND, edgecolor='white', lw=0.8)
axes[2].axvline(2019, color='red', ls='--', lw=1.5, label='Maternity recall start')
axes[2].set_xlabel('Year of Birth', fontsize=11)
axes[2].set_ylabel('Number of Births', fontsize=11)
axes[2].set_title('Birth Year Distribution (bidx=1)\n(Maternity data only available 2019+)',
                   fontsize=12, fontweight='bold')
axes[2].legend(fontsize=9)

plt.suptitle('Outcome Overview - Infant Mortality, GDHS 2022 (bidx=1)',
             fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('fig_02_outcome_overview.png', dpi=150, bbox_inches='tight')
plt.show()
print("    → fig_02_outcome_overview.png")


# ════════════════════════════════════════════════════════════
# 13. EDA - PROTECTED FEATURES
# ════════════════════════════════════════════════════════════
print("\n[13] EDA - protected features vs outcome …")

# Fig: categorical protected features - death rate bars
# place_of_delivery and education_level removed (MI=0 in bidx=1 sample)
cat_prot = ['sex_of_child','multiple_birth','caesarean_section',
            'wealth_index','marital_status','region']
cat_prot = [c for c in cat_prot if c in df.columns]

fig, axes = plt.subplots(2, 4, figsize=(22, 11))
axes = axes.flatten()
for i, var in enumerate(cat_prot):
    ct = pd.crosstab(df[var], df['infant_death'], normalize='index') * 100
    if 1 in ct.columns:
        ct_sorted = ct[1].sort_values()
        colors_bar = [C_DIED if v > ct[1].mean() else C_SURV
                      for v in ct_sorted.values]
        ct_sorted.plot(kind='barh', ax=axes[i],
                       color=colors_bar, edgecolor='white')
        axes[i].axvline(ct[1].mean(), color='black', ls='--',
                        lw=1, alpha=0.6, label='Mean')
        axes[i].set_title(var.replace('_',' ').title(),
                          fontsize=11, fontweight='bold')
        axes[i].set_xlabel('Death rate (%)', fontsize=9)
        axes[i].set_ylabel('')
        axes[i].legend(fontsize=7)
plt.suptitle('Infant Death Rate by Protected Categorical Features\n(GDHS 2022, bidx=1)',
             fontsize=15, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig('fig_03_protected_categoricals.png', dpi=150, bbox_inches='tight')
plt.show()
print("    → fig_03_protected_categoricals.png")

# Fig: continuous protected features
# mothers_age removed (MI=0 in bidx=1 sample)
cont_prot = ['preceding_birth_interval',
             'pregnancy_duration','total_children_born']
cont_prot = [c for c in cont_prot if c in df_enc.columns]

fig, axes = plt.subplots(1, len(cont_prot), figsize=(5*len(cont_prot), 5))
for i, var in enumerate(cont_prot):
    ax = axes[i]
    d_v = died_e[var].dropna()
    s_v = survived_e[var].dropna()
    ax.hist(s_v, bins=25, density=True, alpha=0.55, color=C_SURV, label='Survived')
    ax.hist(d_v, bins=25, density=True, alpha=0.55, color=C_DIED, label='Died')
    stat, p = mannwhitneyu(d_v, s_v, alternative='two-sided')
    ax.set_title(f"{var.replace('_',' ').title()}\n(p={p:.3f})",
                 fontsize=10, fontweight='bold')
    ax.legend(fontsize=8)
plt.suptitle('Protected Continuous Variables - Distribution by Outcome',
             fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('fig_04_protected_distributions.png', dpi=150, bbox_inches='tight')
plt.show()
print("    → fig_04_protected_distributions.png")

# Fig: antenatal visits (treated as ordered category)
fig, ax = plt.subplots(figsize=(11, 5))
anc_ct = pd.crosstab(df['antenatal_visits'], df['infant_death'], normalize='index') * 100
if 1 in anc_ct.columns:
    anc_ct[1].plot(kind='bar', ax=ax, color=C_DIED, edgecolor='white')
ax.set_xlabel('Antenatal Visits', fontsize=12)
ax.set_ylabel('Infant Death Rate (%)', fontsize=12)
ax.set_title('Infant Death Rate by Antenatal Visits\n'
             '(not_in_recall_window = birth before 3-yr window)',
             fontsize=13, fontweight='bold')
ax.tick_params(axis='x', rotation=45)
plt.tight_layout()
plt.savefig('fig_05_antenatal_visits.png', dpi=150, bbox_inches='tight')
plt.show()
print("    → fig_05_antenatal_visits.png")


# ════════════════════════════════════════════════════════════
# 14. UNIVARIATE ANALYSIS
# ════════════════════════════════════════════════════════════
print("\n[14] Univariate analysis (Chi-Square / Mann-Whitney) …")

results_uni = []
for var in ALL_PREDS:
    if var not in df_enc.columns: continue
    n_uniq = df_enc[var].nunique(dropna=True)
    try:
        if n_uniq <= 12 or var in obj_cols:
            chi2, p, _, _ = chi2_contingency(
                pd.crosstab(df_enc[var], df_enc['infant_death']))
            results_uni.append({'Variable': var, 'Test': 'Chi-Square',
                                'Statistic': round(chi2,4), 'P_Value': round(p,6)})
        else:
            stat, p = mannwhitneyu(died_e[var].dropna(),
                                   survived_e[var].dropna(),
                                   alternative='two-sided')
            results_uni.append({'Variable': var, 'Test': 'Mann-Whitney',
                                'Statistic': round(stat,4), 'P_Value': round(p,6)})
    except Exception:
        pass

uni_df = pd.DataFrame(results_uni).sort_values('P_Value')
sig05  = uni_df[uni_df['P_Value'] < 0.05]['Variable'].tolist()
sig10  = uni_df[uni_df['P_Value'] < 0.10]['Variable'].tolist()

print(f"    Variables tested       : {len(uni_df)}")
print(f"    Significant p<0.05     : {len(sig05)}")
print(f"    Significant p<0.10     : {len(sig10)}")
print("\n    Top 30 by p-value:")
print(uni_df.head(30)[['Variable','Test','P_Value']].to_string(index=False))

# Fig: volcano plot
log_p    = -np.log10(uni_df['P_Value'].clip(lower=1e-300))
c_vol    = [C_DIED if p < 0.05 else C_GREY for p in uni_df['P_Value']]
prot_idx = [i for i, v in enumerate(uni_df['Variable']) if v in PROT_NAMES]

fig, ax = plt.subplots(figsize=(12, 7))
ax.scatter(range(len(uni_df)), log_p, c=c_vol, s=40, alpha=0.7, zorder=2)
ax.scatter(prot_idx, log_p.iloc[prot_idx],
           c=C_PROT, s=90, zorder=3, marker='D', label='Protected feature')
ax.axhline(-np.log10(0.05), color='red', ls='--', lw=1.2, label='p = 0.05')
ax.axhline(-np.log10(0.10), color=C_WARN, ls=':', lw=1.2, label='p = 0.10')
ax.set_xlabel('Variable Rank (sorted by p-value)', fontsize=12)
ax.set_ylabel('–log₁₀(p-value)', fontsize=12)
ax.set_title('Univariate Significance - All Candidate Variables vs Infant Death\n'
             '(GDHS 2022, bidx=1 restricted)',
             fontsize=14, fontweight='bold')
ax.legend(fontsize=10)
plt.tight_layout()
plt.savefig('fig_06_volcano_univariate.png', dpi=150, bbox_inches='tight')
plt.show()
print("    → fig_06_volcano_univariate.png")



# ════════════════════════════════════════════════════════════
# 15. MULTICOLLINEARITY CHECK + DROP REDUNDANT VARIABLES
# ════════════════════════════════════════════════════════════
print("\n[15] Multicollinearity check - flagging and dropping redundant variables …")

num_all = df_enc.select_dtypes(include=[np.number]).columns.tolist()
num_all = [c for c in num_all if c not in ('cluster_id','strata_id')]
corr_full = df_enc[num_all].corr()

high_pairs = []
for i in range(len(corr_full.columns)):
    for j in range(i+1, len(corr_full.columns)):
        r = corr_full.iloc[i,j]
        if abs(r) > 0.70:
            high_pairs.append((corr_full.columns[i], corr_full.columns[j], round(r,3)))
high_pairs.sort(key=lambda x: abs(x[2]), reverse=True)

print(f"    Pairs with |r| > 0.70: {len(high_pairs)}")
for a, b, r in high_pairs:
    flag = "  ← involves PROTECTED var" if (a in PROT_NAMES or b in PROT_NAMES) else ""
    print(f"      {a:<30} ↔ {b:<30} r = {r}{flag}")

# ── Drop the redundant member of each correlated pair ────────
# Rule: if both are candidates, drop the re-encoding / derivative /
#       lower-interpretability member.
#       If one is PROTECTED, always drop the candidate - never the
#       protected variable.
MULTICOLLINEAR_DROP = [
    # r = ±1.0  perfect duplicates
    'v3a00z',   # r=-1.0  with v3a00y  (any-source = complement of no-source)
    'v616',     # r= 1.0  with v603    (preferred wait time grouped; keep v603 raw)
    'v509',     # r= 0.999 with v508   (CMC date = numeric re-encoding of v508)
    'v512',     # r=-0.999 with v508   (years-since = arithmetic mirror of v508)
    # r > 0.95
    'v461',     # r= 0.997 with ml101  (slept under bednet - exact duplicate; keep ml101)
    'v481e',    # r= 0.983 with v481   (NHIS sub-type is subset of v481 health insurance)
    'v731',     # r=-0.967 with v714   (worked last 12 m ≈ currently working; keep v714)
    # r > 0.80
    'v226',     # r=-0.923 with v216   (time since period - keep v216 amenorrhea flag)
    'v044',     # r= 0.796 with v042   (selected-for-DV-module admin flag; keep v042)
    'v171b',    # r=-0.910 with v171a  (frequency of internet - keep v171a binary)
    'v245',     # r= 0.824 with v228   (pregnancy losses ≈ terminated; keep v228)
    'v543a',    # r= 0.801 with v542   (church cert type ⊂ v542 has-certificate)
    'v604',     # r= 0.736 with v603   (grouped waiting time - derived; keep v603)
    'v624',     # r= 0.762 with v364   (unmet-need composite - derived; keep v364)
    'v169c',    # r= 0.759 with v171a  (smartphone ≈ internet use; keep v171a)
    'v151',     # r= 0.744 with v150   (sex of HH head ~ relationship; keep v150)
    'v628',     # r= 0.715 with v627   (ideal daughters ≈ ideal sons; keep v627)
    'v380',     # r= 0.706 with v379   (source-known sub-type; keep v379)
    # Correlated with maternal age - still drop to reduce redundancy
    'v508',     # r=-0.769 with maternal age (year first cohabitation; age-collinear)
]

# Safety guard - never drop a protected variable
MULTICOLLINEAR_DROP = [c for c in MULTICOLLINEAR_DROP
                        if c not in PROT_NAMES and c in df_enc.columns]

df     = df.drop(columns=MULTICOLLINEAR_DROP, errors='ignore')
df_enc = df_enc.drop(columns=MULTICOLLINEAR_DROP, errors='ignore')
ALL_PREDS = [c for c in df_enc.columns
             if c not in ('infant_death', 'cluster_id', 'strata_id')]

print(f"\n    Dropped {len(MULTICOLLINEAR_DROP)} redundant variables from correlated pairs")
print(f"    Updated predictor pool: {len(ALL_PREDS)}")
assert all(v in df_enc.columns for v in PROT_NAMES), "ERROR: protected variable was dropped!"
print("    Protected features intact: ✓")

# Refresh sig05 / sig10 on the updated predictor list so downstream
# MI and BIC steps see the clean, non-redundant pool
uni_df_upd = uni_df[uni_df['Variable'].isin(ALL_PREDS)]
sig05 = uni_df_upd[uni_df_upd['P_Value'] < 0.05]['Variable'].tolist()
sig10 = uni_df_upd[uni_df_upd['P_Value'] < 0.10]['Variable'].tolist()

# Recompute numeric list after drops
num_all = df_enc.select_dtypes(include=[np.number]).columns.tolist()
num_all = [c for c in num_all if c not in ('cluster_id','strata_id')]

# ── Heatmap: protected numerics + top sig numeric candidates ─
# mothers_age removed from focal (MI=0 in bidx=1 sample)
focal = [c for c in ['preceding_birth_interval',
                      'pregnancy_duration',
                      'total_children_born','infant_death']
         if c in df_enc.columns]
extra = [v for v in sig05 if v in num_all and v not in focal][:8]
hm_vars = focal + extra

corr_sub = df_enc[hm_vars].corr()

# The lower-triangle display works by masking the upper triangle (k=1 keeps
# the diagonal visible). However if a variable appears early in the list its
# row only has cells from col-0 up to the diagonal - which is correct.
# The blank-row issue in the previous version was caused by birth_order and
# preceding_birth_interval having near-zero variance after label encoding,
# making seaborn render their cells as white. Fix: reorder so the most
# correlated / informative variables come first, pushing sparse vars to the
# bottom where they still show their diagonal cell + cells below them.
# Move preceding_birth_interval to the end so it appears in a lower row
# and shows all its correlation cells (lower-triangle geometry).
# birth_order is already removed from PROTECTED and focal.
hm_vars_ordered = (
    [v for v in hm_vars if v != 'preceding_birth_interval']
    + (['preceding_birth_interval'] if 'preceding_birth_interval' in hm_vars else [])
)
corr_sub = df_enc[hm_vars_ordered].corr()

mask = np.triu(np.ones_like(corr_sub, dtype=bool), k=1)  # hide above diagonal only

fig, ax = plt.subplots(figsize=(13, 11))
sns.heatmap(corr_sub, annot=True, fmt='.2f', cmap='coolwarm',
            center=0, square=True, linewidths=0.5,
            mask=mask, ax=ax, annot_kws={'size': 8},
            vmin=-1, vmax=1, cbar_kws={'shrink': 0.8})
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', fontsize=9)
ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=9)
ax.set_title('Correlation Heatmap - Protected + Top Numeric Candidates\n(After Multicollinearity Pruning | GDHS 2022)',
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('fig_07_correlation_heatmap.png', dpi=150, bbox_inches='tight')
plt.show()
print("    → fig_07_correlation_heatmap.png")


# ════════════════════════════════════════════════════════════
# 16. MUTUAL INFORMATION
# ════════════════════════════════════════════════════════════
print("\n[16] Mutual Information (MI) scoring …")

X_mi = df_enc[ALL_PREDS]
y_mi = df_enc['infant_death']
mi_scores = mutual_info_classif(X_mi, y_mi, discrete_features='auto',
                                 random_state=42, n_neighbors=5)
mi_df = pd.DataFrame({'Variable': X_mi.columns,
                       'MI_Score': mi_scores}
                     ).sort_values('MI_Score', ascending=False)
mi_df['Protected'] = mi_df['Variable'].isin(PROT_NAMES)
mi_nonzero = mi_df[mi_df['MI_Score'] > 0]

print(f"    Variables tested   : {len(mi_df)}")
print(f"    MI > 0             : {len(mi_nonzero)}")
print("\n    Top 30 MI Scores:")
print(mi_df.head(30).to_string(index=False))

# Fig: MI bar chart top 30
top30_mi = mi_df.head(30)
c_mi = [C_PROT if p else C_CAND for p in top30_mi['Protected']]
fig, ax = plt.subplots(figsize=(11, 9))
ax.barh(top30_mi['Variable'].values[::-1],
        top30_mi['MI_Score'].values[::-1],
        color=c_mi[::-1], edgecolor='white', lw=0.8)
ax.set_xlabel('Mutual Information Score', fontsize=12)
ax.set_title('Mutual Information - Top 30 Predictors of Infant Mortality\n'
             '(Purple = Protected | Blue = New candidate | GDHS 2022)',
             fontsize=13, fontweight='bold')
ax.legend(handles=[Patch(color=C_PROT, label='Protected'),
                   Patch(color=C_CAND, label='New candidate')], fontsize=10)
plt.tight_layout()
plt.savefig('fig_08_mutual_information.png', dpi=150, bbox_inches='tight')
plt.show()
print("    → fig_08_mutual_information.png")


# ════════════════════════════════════════════════════════════
# 17. PCA
# ════════════════════════════════════════════════════════════
print("\n[17] Principal Component Analysis …")

pca_vars = df_enc[ALL_PREDS].select_dtypes(include=[np.number]).columns.tolist()
sc_pca   = StandardScaler()
X_pca_sc = sc_pca.fit_transform(df_enc[pca_vars])
pca_m    = PCA(random_state=42)
pca_m.fit(X_pca_sc)

expl    = pca_m.explained_variance_ratio_
cumexpl = np.cumsum(expl)
n90     = np.argmax(cumexpl >= 0.90) + 1
n95     = np.argmax(cumexpl >= 0.95) + 1
print(f"    Components for 90% variance: {n90}")
print(f"    Components for 95% variance: {n95}")

loadings_df = pd.DataFrame(
    np.abs(pca_m.components_[:5].T),
    index=pca_vars,
    columns=[f'PC{i+1}' for i in range(5)]
)
print("\n    Top 15 PC1 loadings:")
print(loadings_df['PC1'].sort_values(ascending=False).head(15).to_string())

# Fig: scree + cumulative + biplot
fig = plt.figure(figsize=(18, 5))
ax1 = fig.add_subplot(1, 3, 1)
ax2 = fig.add_subplot(1, 3, 2)
ax3 = fig.add_subplot(1, 3, 3)

n_show = min(40, len(expl))
ax1.bar(range(1, n_show+1), expl[:n_show]*100, color=C_CAND, edgecolor='white')
ax1.set_xlabel('Principal Component', fontsize=11)
ax1.set_ylabel('Explained Variance (%)', fontsize=11)
ax1.set_title('Scree Plot', fontsize=12, fontweight='bold')

ax2.plot(range(1, len(cumexpl)+1), cumexpl*100, color=C_CAND, lw=2)
ax2.axhline(90, color=C_WARN, ls='--', lw=1.5, label=f'90% ({n90} PCs)')
ax2.axhline(95, color=C_DIED, ls='--', lw=1.5, label=f'95% ({n95} PCs)')
ax2.set_xlabel('Number of Components', fontsize=11)
ax2.set_ylabel('Cumulative Variance (%)', fontsize=11)
ax2.set_title('Cumulative Variance Explained', fontsize=12, fontweight='bold')
ax2.legend(fontsize=10)

X2d   = pca_m.transform(X_pca_sc)[:, :2]
c_bi  = [C_DIED if y == 1 else C_SURV for y in df_enc['infant_death']]
ax3.scatter(X2d[:,0], X2d[:,1], c=c_bi, alpha=0.25, s=10)
ax3.legend(handles=[Patch(color=C_SURV, label='Survived'),
                    Patch(color=C_DIED, label='Died')], fontsize=10)
ax3.set_xlabel(f'PC1 ({expl[0]*100:.1f}%)', fontsize=11)
ax3.set_ylabel(f'PC2 ({expl[1]*100:.1f}%)', fontsize=11)
ax3.set_title('Biplot - PC1 vs PC2 by Outcome', fontsize=12, fontweight='bold')

plt.suptitle('PCA - Dimensionality Structure (GDHS 2022, bidx=1)',
             fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('fig_09_pca.png', dpi=150, bbox_inches='tight')
plt.show()
print("    → fig_09_pca.png")


# ════════════════════════════════════════════════════════════
# 18. BIC FORWARD STEPWISE LOGISTIC REGRESSION
# ════════════════════════════════════════════════════════════
print("\n[18] BIC forward stepwise logistic regression …")

# Pool = protected ∪ top-50 MI-nonzero ∪ sig at p<0.10
bic_pool = list(set(
    PROT_NAMES
    + mi_nonzero['Variable'].head(50).tolist()
    + sig10[:60]
))
bic_pool = [v for v in bic_pool
            if v in df_enc.columns
            and v not in ('infant_death','cluster_id','strata_id')]

# Scale numerics
df_bic = df_enc[bic_pool + ['infant_death']].copy()
num_bic = df_bic[bic_pool].select_dtypes(include=[np.number]).columns.tolist()
sc_bic  = StandardScaler()
df_bic[num_bic] = sc_bic.fit_transform(df_bic[num_bic])

X_bic = df_bic[bic_pool].values
y_bic = df_bic['infant_death'].values
N     = len(y_bic)

def bic_logistic(X_sel, y, n):
    if X_sel.shape[1] == 0: return np.inf
    m = LogisticRegression(max_iter=5000, C=1e6, solver='lbfgs',
                            random_state=42, tol=1e-4)
    m.fit(X_sel, y)
    p = np.clip(m.predict_proba(X_sel)[:,1], 1e-15, 1-1e-15)
    ll = np.sum(y*np.log(p) + (1-y)*np.log(1-p))
    k  = X_sel.shape[1] + 1
    return -2*ll + k*np.log(n)

sel_idx  = []
rem_idx  = list(range(len(bic_pool)))
cur_bic  = np.inf
bic_hist = []
MAX_STEP = 30

print(f"    Pool: {len(bic_pool)} variables. Forward stepwise …")
for step in range(MAX_STEP):
    best_bic = cur_bic
    best_add = None
    for idx in rem_idx:
        b = bic_logistic(X_bic[:, sel_idx + [idx]], y_bic, N)
        if b < best_bic:
            best_bic, best_add = b, idx
    if best_add is None:
        print(f"    Converged at step {step+1} (no BIC improvement)")
        break
    sel_idx.append(best_add)
    rem_idx.remove(best_add)
    cur_bic = best_bic
    vname   = bic_pool[best_add]
    bic_hist.append({'Step': step+1, 'Variable': vname, 'BIC': round(best_bic, 2)})
    print(f"      Step {step+1:2d}: + {vname:<40} BIC = {best_bic:.2f}")

bic_selected = [bic_pool[i] for i in sel_idx]
bic_hist_df  = pd.DataFrame(bic_hist)
print(f"\n    BIC stepwise selected: {len(bic_selected)} variables")

# Fig: BIC path
bic_delta = [bic_hist_df['BIC'].iloc[0] - bic_hist_df['BIC'].iloc[i]
             for i in range(len(bic_hist_df))]
c_bic = [C_PROT if v in PROT_NAMES else C_CAND
         for v in bic_hist_df['Variable']]

fig, axes = plt.subplots(1, 2, figsize=(15, 6))
axes[0].plot(bic_hist_df['Step'], bic_hist_df['BIC'],
             marker='o', color=C_WARN, lw=2, markersize=6)
axes[0].set_xlabel('Step (variables added)', fontsize=12)
axes[0].set_ylabel('BIC', fontsize=12)
axes[0].set_title('BIC Reduction Path\nForward Stepwise Logistic Regression',
                   fontsize=13, fontweight='bold')
axes[0].grid(alpha=0.3)

axes[1].barh(bic_hist_df['Variable'].values[::-1],
             bic_delta[::-1], color=c_bic[::-1], edgecolor='white')
axes[1].set_xlabel('Cumulative BIC Reduction', fontsize=12)
axes[1].set_title('BIC Contribution per Variable\n(Purple = Protected | Blue = New discovery)',
                   fontsize=13, fontweight='bold')
axes[1].legend(handles=[Patch(color=C_PROT, label='Protected'),
                         Patch(color=C_CAND, label='New candidate')], fontsize=10)
plt.suptitle('BIC-Stepwise Variable Selection (GDHS 2022, bidx=1)',
             fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('fig_10_bic_stepwise.png', dpi=150, bbox_inches='tight')
plt.show()
print("    → fig_10_bic_stepwise.png")


# ════════════════════════════════════════════════════════════
# 19. CONSENSUS VARIABLE SELECTION
# ════════════════════════════════════════════════════════════
print("\n[19] Consensus variable selection (voting across 3 criteria) …")

votes = {v: 0 for v in ALL_PREDS if v in df_enc.columns}
for v in sig05:
    if v in votes: votes[v] += 1
for v in mi_nonzero['Variable']:
    if v in votes: votes[v] += 1
for v in bic_selected:
    if v in votes: votes[v] += 1

votes_df = pd.DataFrame({'Variable': list(votes.keys()),
                          'Votes': list(votes.values())})
votes_df['Protected'] = votes_df['Variable'].isin(PROT_NAMES)
votes_df = votes_df.sort_values(['Protected','Votes'], ascending=[False,False])

# OPTIMISED SELECTION: 10 protected + top 13 new candidates = 23 total
# Target: EPV >= 10  →  237 deaths / 23 vars = 10.3 EPV
# New candidates ranked purely by MI score (most direct measure of
# predictive signal for the outcome). Votes used only as a tiebreaker.
# mothers_age, education_level, place_of_delivery removed from PROTECTED
# above because MI = 0 in this bidx=1 restricted dataset.

TARGET_TOTAL = 23
N_PROTECTED  = len(PROT_NAMES)          # 10
N_NEW        = TARGET_TOTAL - N_PROTECTED  # 13

# Rank all non-protected candidates by MI score descending
cands_ranked = votes_df[~votes_df['Protected']].copy()
cands_ranked = cands_ranked.merge(
    mi_df[['Variable','MI_Score']], on='Variable', how='left'
)
cands_ranked['MI_Score'] = cands_ranked['MI_Score'].fillna(0)
cands_ranked = cands_ranked.sort_values(
    ['MI_Score','Votes'], ascending=[False, False]
)

new_vars   = cands_ranked['Variable'].head(N_NEW).tolist()
final_vars = (
    votes_df[votes_df['Protected']]['Variable'].tolist()
    + new_vars
)

epv = 237 / len(final_vars)
print(f"    Protected retained (MI>0 only)          : {N_PROTECTED}")
print(f"    New candidates selected (top {N_NEW} by MI)   : {len(new_vars)}")
print(f"    Total final variables                   : {len(final_vars)}")
print(f"    EPV (237 / {len(final_vars)} vars)                   : {epv:.1f}  {'✓ OK' if epv>=10 else '⚠ below 10'}")
print("\n    ── New variables selected (ranked by MI score) ──")
for v in new_vars:
    row    = cands_ranked[cands_ranked['Variable'] == v].iloc[0]
    in_bic = '✓ BIC' if v in bic_selected else '  '
    print(f"      {v:<30} MI={row['MI_Score']:.5f}  votes={int(row['Votes'])}  {in_bic}")

# Fig: consensus vote chart
top40 = votes_df.head(40)
c_vt  = [C_PROT if p else C_CAND for p in top40['Protected']]
fig, ax = plt.subplots(figsize=(10, 12))
bars_v = ax.barh(top40['Variable'].values[::-1],
                  top40['Votes'].values[::-1],
                  color=c_vt[::-1], edgecolor='white')
ax.axvline(2, color='red', ls='--', lw=1.5, label='Selection threshold (≥2)')
ax.set_xlabel('Number of Criteria Met (max 3)', fontsize=12)
ax.set_xlim(0, 3.5)
ax.bar_label(bars_v, fmt='%.0f', fontsize=9, padding=2)
ax.set_title('Consensus Variable Selection\n'
             'MI + Univariate + BIC | GDHS 2022, bidx=1',
             fontsize=13, fontweight='bold')
ax.legend(handles=[
    Patch(color=C_PROT, label='Protected feature'),
    Patch(color=C_CAND, label='New candidate'),
    plt.Line2D([0],[0], color='red', ls='--', label='Threshold ≥2')
], fontsize=10)
plt.tight_layout()
plt.savefig('fig_11_variable_votes.png', dpi=150, bbox_inches='tight')
plt.show()
print("    → fig_11_variable_votes.png")


# ════════════════════════════════════════════════════════════
# 19b. CORRELATION FILTER - NEW CANDIDATES ONLY
#      Drop one variable per correlated pair (|r| ≥ 0.70)
#      among the new (non-protected) candidates that passed
#      consensus voting.  The variable with the LOWER vote
#      count is dropped; ties broken by lower MI score.
# ════════════════════════════════════════════════════════════
print("\n[19b] Correlation filter on new candidates (|r| ≥ 0.70) …")

CORR_THRESH = 0.70

if len(new_vars) >= 2:
    # ── Spearman correlation matrix for new candidates only ──
    corr_matrix = df_enc[new_vars].corr(method='spearman').abs()

    # ── Build a lookup: variable → votes ─────────────────────
    votes_lookup = votes_df.set_index('Variable')['Votes'].to_dict()

    # ── Build a lookup: variable → MI score ──────────────────
    # mi_nonzero was computed in section [16]; use it as tiebreaker
    mi_lookup = mi_nonzero.set_index('Variable')['MI'].to_dict() \
        if 'Variable' in mi_nonzero.columns and 'MI' in mi_nonzero.columns \
        else {}

    # ── Identify pairs above threshold (upper triangle only) ─
    corr_pairs = []
    for i in range(len(new_vars)):
        for j in range(i + 1, len(new_vars)):
            r = corr_matrix.iloc[i, j]
            if r >= CORR_THRESH:
                corr_pairs.append((new_vars[i], new_vars[j], round(r, 4)))

    print(f"    Correlated pairs found (|r| ≥ {CORR_THRESH}): {len(corr_pairs)}")

    to_drop_corr = set()
    for v1, v2, r_val in corr_pairs:
        # Skip if one is already scheduled for removal
        if v1 in to_drop_corr or v2 in to_drop_corr:
            continue
        votes_v1 = votes_lookup.get(v1, 0)
        votes_v2 = votes_lookup.get(v2, 0)
        mi_v1    = mi_lookup.get(v1, 0)
        mi_v2    = mi_lookup.get(v2, 0)
        # Drop the lower-votes variable; use MI as tiebreaker
        if votes_v1 < votes_v2:
            loser = v1
        elif votes_v2 < votes_v1:
            loser = v2
        else:
            loser = v1 if mi_v1 <= mi_v2 else v2
        keeper = v2 if loser == v1 else v1
        to_drop_corr.add(loser)
        print(f"      Pair  : {v1} ↔ {v2}  |r|={r_val}")
        print(f"      Keep  : {keeper}  (votes={votes_lookup.get(keeper,0)}, MI={mi_lookup.get(keeper,0):.4f})")
        print(f"      Drop  : {loser}   (votes={votes_lookup.get(loser,0)}, MI={mi_lookup.get(loser,0):.4f})")

    if to_drop_corr:
        new_vars   = [v for v in new_vars   if v not in to_drop_corr]
        final_vars = (
            votes_df[votes_df['Protected']]['Variable'].tolist()
            + new_vars
        )
        print(f"\n    Dropped {len(to_drop_corr)} correlated new candidate(s): {sorted(to_drop_corr)}")
    else:
        print("    No variables dropped - no pairs exceeded the threshold.")

    print(f"    New candidates after correlation filter : {len(new_vars)}")
    print(f"    Total final variables (updated)         : {len(final_vars)}")
else:
    print("    Fewer than 2 new candidates - correlation check skipped.")


# ════════════════════════════════════════════════════════════
# 20. EDA - NEWLY DISCOVERED VARIABLES
# ════════════════════════════════════════════════════════════
print("\n[20] EDA - newly discovered variables …")
if len(new_vars) >= 2:
    n_plot = min(12, len(new_vars))
    n_rows = (n_plot + 2) // 3
    fig, axes = plt.subplots(n_rows, 3, figsize=(18, n_rows*4 + 1))
    axes = axes.flatten()
    for i, var in enumerate(new_vars[:n_plot]):
        ax = axes[i]
        n_uniq = df_enc[var].nunique(dropna=True)
        if n_uniq <= 12:
            ct = pd.crosstab(df_enc[var], df_enc['infant_death'],
                             normalize='index') * 100
            if 1 in ct.columns:
                ct[1].sort_values().plot(kind='bar', ax=ax,
                                          color=C_WARN, edgecolor='white')
                ax.set_title(f'{var}\n(% infant death)', fontsize=10, fontweight='bold')
                ax.tick_params(axis='x', rotation=45)
        else:
            ax.hist(survived_e[var].dropna(), bins=20, density=True,
                    alpha=0.5, color=C_SURV, label='Survived')
            ax.hist(died_e[var].dropna(), bins=20, density=True,
                    alpha=0.5, color=C_DIED, label='Died')
            ax.set_title(var, fontsize=10, fontweight='bold')
            ax.legend(fontsize=7)
    for j in range(i+1, len(axes)):
        axes[j].set_visible(False)
    plt.suptitle('Newly Discovered Variables - Distribution by Outcome\n'
                 '(GDHS 2022, bidx=1)',
                 fontsize=14, fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.savefig('fig_12_new_variables_eda.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("    → fig_12_new_variables_eda.png")


# ════════════════════════════════════════════════════════════
# 21. SAMPLE SIZE JUSTIFICATION
# ════════════════════════════════════════════════════════════
print("\n[21] Sample size justification …")
prev   = df['infant_death'].mean()
n_pred = len(final_vars)
z_a    = sp_stats.norm.ppf(0.975)
z_b    = sp_stats.norm.ppf(0.80)
n_coch = (z_a**2 * prev * (1-prev)) / (0.05**2)
epv_n  = int(np.ceil((n_pred * 10) / prev))
p2     = prev * 0.75
pbar   = (prev + p2) / 2
n_pow  = ((z_a*np.sqrt(2*pbar*(1-pbar)) + z_b*np.sqrt(prev*(1-prev)+p2*(1-p2)))**2
          / (prev-p2)**2)

print(f"\n    Outcome prevalence          : {prev*100:.2f}%")
print(f"    Final predictors            : {n_pred}")
print(f"    Cochran formula             : {int(np.ceil(n_coch)):,}")
print(f"    EPV rule (min n required)   : {epv_n:,}")
print(f"    Power-based (80% power)     : {int(np.ceil(n_pow)):,}")
print(f"    Actual sample (bidx=1)      : {len(df):,}")
print(f"    Actual deaths               : {df['infant_death'].sum()}")
print(f"    EPV achieved                : {df['infant_death'].sum()/n_pred:.1f}")

if df['infant_death'].sum() / n_pred < 10:
    print("\n    ⚠  EPV < 10. Consider: (a) reducing final predictors,")
    print("       (b) using penalised regression (LASSO/Ridge) in Notebook 2,")
    print("       (c) reporting bootstrap-corrected estimates.")


# ════════════════════════════════════════════════════════════
# 22. SAVE CLEAN DATASET
# ════════════════════════════════════════════════════════════
print("\n[22] Saving df_full_clean.csv …")
save_cols = final_vars + ['cluster_id','strata_id','infant_death']
save_cols = [c for c in dict.fromkeys(save_cols) if c in df_enc.columns]
df_full_clean = df_enc[save_cols].copy()
df_full_clean.to_csv('df_full_clean.csv', index=False)
print(f"    Saved. Shape: {df_full_clean.shape}")
print(f"    Columns: {list(df_full_clean.columns)}")


# ════════════════════════════════════════════════════════════
print("\n" + SEP)
print("  ✓  EDA AND FEATURE SELECTION COMPLETE")
print(SEP)
print("""
  Output figures:
    fig_00_imr_selection_bias.png     ← IMR transparency visual
    fig_01_missingness_overview.png
    fig_02_outcome_overview.png
    fig_03_protected_categoricals.png
    fig_04_protected_distributions.png
    fig_05_antenatal_visits.png
    fig_06_volcano_univariate.png
    fig_07_correlation_heatmap.png
    fig_08_mutual_information.png
    fig_09_pca.png
    fig_10_bic_stepwise.png
    fig_11_variable_votes.png
    fig_12_new_variables_eda.png

  Data output  : df_full_clean.csv
""")
