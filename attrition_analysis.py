# =============================================================================
# EMPLOYEE ATTRITION ANALYSIS
# File: attrition_analysis.py
# Description: Exploratory and diagnostic attrition analysis of BOTH voluntary
#              and involuntary turnover — analysed in isolation and combined —
#              identifying patterns and potential drivers of attrition.
# =============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from scipy import stats
from scipy.stats import chi2_contingency, mannwhitneyu
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
import warnings

warnings.filterwarnings('ignore')

# ---------------------------------------------------------------------------
# Global chart style
# ---------------------------------------------------------------------------
sns.set_theme(style='whitegrid', palette='muted', font_scale=1.05)
plt.rcParams.update({
    'figure.dpi': 150,
    'axes.titleweight': 'bold',
    'axes.titlesize': 13,
    'axes.labelsize': 11,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.constrained_layout.use': True,
})

BRAND_BLUE   = '#1F4E79'
BRAND_ORANGE = '#C55A11'
BRAND_GREEN  = '#375623'
PALETTE_2    = [BRAND_BLUE, BRAND_ORANGE]

# Consistent colour per cohort so charts read the same across sections
COHORT_COLOUR = {
    'All Leavers': BRAND_GREEN,
    'Voluntary'  : BRAND_BLUE,
    'Involuntary': BRAND_ORANGE,
}

def slug(name):
    """Filesystem-safe short tag for a cohort name (e.g. 'All Leavers' -> 'all')."""
    return name.lower().split()[0]

FILE_PATH = 'Atrrition Data.xlsx'   # update path as required


# =============================================================================
# 1. DATA PREPARATION
# =============================================================================
print('\n' + '='*65)
print('SECTION 1 – DATA PREPARATION')
print('='*65)

# ── 1.1  Load both sheets ────────────────────────────────────────────────────
turnover_raw   = pd.read_excel(FILE_PATH, sheet_name='Turnover')
headcount_raw  = pd.read_excel(FILE_PATH, sheet_name='Headcount')

# ── 1.2  Normalise column names ──────────────────────────────────────────────
def clean_cols(df):
    df = df.copy()
    df.columns = (
        df.columns
          .str.strip()
          .str.replace(r'[^\w\s]', '', regex=True)
          .str.replace(r'\s+', '_', regex=True)
          .str.title()
    )
    return df

turnover  = clean_cols(turnover_raw)
headcount = clean_cols(headcount_raw)

print('\nTurnover columns :', turnover.columns.tolist())
print('Headcount columns:', headcount.columns.tolist())
print('\nTurnover shape :', turnover.shape)
print('Headcount shape:', headcount.shape)

# ── 1.3  Standardise key column references ───────────────────────────────────
COL = {
    'category'         : 'Offboard_Category',
    'reason'           : 'Offboard_Reason',
    'age'              : 'Age',
    'age_band'         : 'Age_Band',
    'tenure'           : 'Tenure_Years',
    'tenure_band'      : 'Tenure_Band',
    'tenure_sort'      : 'Tenure_Sort_Order',   # provided by source data – use for ordering, not text
    'age_sort'         : 'Age_Sort_Order',      # provided by source data – use for ordering, not text
    'daily_salary'     : 'Daily_Salary',
    'base_pay'         : 'Total_Base_Pay__Amount',   # actual column name in Turnover sheet
    'commute_km'       : 'Commute_Km',
    'commute_band'     : 'Commute_Band',
    'area'             : 'Area',
    'division'         : 'Division_57813066',         # actual column name in Turnover sheet
    'business_unit'    : 'Business_Unit',
    'harrods_band'     : 'Harrods_Band',
    'job_profile'      : 'Job_Profile',
    'term_date'        : 'Termination_Date',
}

# Flexible column-finder (case-insensitive partial match fallback)
def find_col(df, target):
    if target in df.columns:
        return target
    for c in df.columns:
        if target.lower().replace('_', '') in c.lower().replace('_', ''):
            return c
    raise KeyError(f"Cannot find column matching '{target}' in {df.columns.tolist()}")

# Rename to canonical names
rename_map = {}
for canonical, raw_guess in COL.items():
    try:
        found = find_col(turnover, raw_guess)
        rename_map[found] = raw_guess
    except KeyError:
        pass

turnover = turnover.rename(columns=rename_map)

# ── 1.4  Data-type coercion ───────────────────────────────────────────────────
numeric_cols = [COL['age'], COL['tenure'], COL['daily_salary'],
                COL['base_pay'], COL['commute_km']]

for col in numeric_cols:
    if col in turnover.columns:
        turnover[col] = pd.to_numeric(turnover[col], errors='coerce')

if COL['term_date'] in turnover.columns:
    turnover[COL['term_date']] = pd.to_datetime(
        turnover[COL['term_date']], errors='coerce', dayfirst=True
    )

# Derive month from Termination_Date for trend analysis
turnover['Term_Month'] = turnover[COL['term_date']].dt.to_period('M')

# ── 1.5  Missing-value audit ─────────────────────────────────────────────────
missing = turnover.isnull().sum()
missing_pct = (missing / len(turnover) * 100).round(1)
missing_report = pd.DataFrame({'Missing_N': missing, 'Missing_%': missing_pct})
missing_report = missing_report[missing_report['Missing_N'] > 0].sort_values('Missing_%', ascending=False)
print('\nMissing value summary:')
print(missing_report.to_string())

# ── 1.6  Binary target variable ──────────────────────────────────────────────
if COL['category'] in turnover.columns:
    turnover[COL['category']] = turnover[COL['category']].str.strip().str.title()
    turnover['Is_Voluntary'] = (turnover[COL['category']] == 'Voluntary').astype(int)
else:
    raise KeyError(f"Column '{COL['category']}' not found – check sheet/column names.")

# ── 1.7  Early-leaver flags ──────────────────────────────────────────────────
# "Early leaver" is analysed at two thresholds: within 1 year and within 2 years.
# EARLY_LEAVER_THRESHOLDS drives every downstream early-leaver section – add or
# remove thresholds here and Section 5 will pick them up automatically.
EARLY_LEAVER_THRESHOLDS = {
    '<1 Year' : 1,
    '<2 Years': 2,
}

if COL['tenure'] in turnover.columns:
    for label, cutoff in EARLY_LEAVER_THRESHOLDS.items():
        flag_col = f'Early_Leaver_{cutoff}Y'
        turnover[flag_col] = (turnover[COL['tenure']] < cutoff).astype(int)

# ── 1.7b  Ordered tenure band list (uses the source Tenure_Sort_Order column) ─
# Sorting by band TEXT is unreliable ("10-15 Years" would sort before "5+ Years"
# alphabetically). The source data provides Tenure_Sort_Order for exactly this
# purpose, so we derive the correct <1yr -> 15+yr ordering from it directly.
TENURE_BAND_ORDER = []
if COL['tenure_band'] in turnover.columns and COL['tenure_sort'] in turnover.columns:
    TENURE_BAND_ORDER = (
        turnover[[COL['tenure_band'], COL['tenure_sort']]]
        .dropna()
        .drop_duplicates()
        .sort_values(COL['tenure_sort'])
        [COL['tenure_band']]
        .tolist()
    )
elif COL['tenure_band'] in turnover.columns:
    # Fallback if Tenure_Sort_Order isn't present: alphabetical (not ideal)
    TENURE_BAND_ORDER = sorted(turnover[COL['tenure_band']].dropna().unique().tolist())

# ── 1.8  Define analysis cohorts ─────────────────────────────────────────────
# Every deep-dive section runs across these three cohorts so that voluntary and
# involuntary turnover are each examined in isolation AND combined.
COHORTS = {
    'All Leavers': turnover.copy(),
    'Voluntary'  : turnover[turnover['Is_Voluntary'] == 1].copy(),
    'Involuntary': turnover[turnover['Is_Voluntary'] == 0].copy(),
}

# Convenience handles retained for the comparative sections
vol   = COHORTS['Voluntary']
invol = COHORTS['Involuntary']

print('\nData preparation complete.')
print(f'  Total records  : {len(turnover):,}')
for name, df in COHORTS.items():
    print(f'  {name:<12} : {len(df):,}')


# =============================================================================
# 2. HIGH-LEVEL ATTRITION SUMMARY
# =============================================================================
print('\n' + '='*65)
print('SECTION 2 – HIGH-LEVEL ATTRITION SUMMARY')
print('='*65)

total         = len(turnover)
total_vol     = turnover['Is_Voluntary'].sum()
total_invol   = total - total_vol
pct_vol       = total_vol / total * 100
pct_invol     = total_invol / total * 100

print(f'\n  Total leavers        : {total:,}')
print(f'  Voluntary leavers    : {total_vol:,}  ({pct_vol:.1f}%)')
print(f'  Involuntary leavers  : {total_invol:,}  ({pct_invol:.1f}%)')

# ── 2a  Donut + monthly trend ────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(15, 5))

wedges, texts, autotexts = axes[0].pie(
    [total_vol, total_invol],
    labels=['Voluntary', 'Involuntary'],
    autopct='%1.1f%%',
    colors=PALETTE_2,
    startangle=90,
    wedgeprops={'width': 0.55, 'edgecolor': 'white', 'linewidth': 2},
    textprops={'fontsize': 10},
)
for at in autotexts:
    at.set_fontweight('bold')
axes[0].set_title('Voluntary vs Involuntary Attrition')

# Monthly attrition trend (derived from Termination_Date), split by category
monthly = (
    turnover.groupby(['Term_Month', COL['category']])
            .size()
            .reset_index(name='Count')
)
monthly['Term_Month'] = monthly['Term_Month'].astype(str)
monthly_pivot = monthly.pivot(
    index='Term_Month', columns=COL['category'], values='Count'
).fillna(0).sort_index()

monthly_pivot.plot(
    kind='bar', ax=axes[1], color=PALETTE_2, edgecolor='white', linewidth=0.5
)
axes[1].set_title('Monthly Attrition Trend')
axes[1].set_xlabel('Month')
axes[1].set_ylabel('Number of Leavers')
axes[1].tick_params(axis='x', rotation=45)
axes[1].legend(title='Category', frameon=False)

fig.suptitle('Section 2 – High-Level Attrition Summary', fontweight='bold', fontsize=14)
plt.savefig('s2_high_level_summary.png', bbox_inches='tight')
plt.show()

# ── 2b  Top off-board reasons for EACH cohort ────────────────────────────────
if COL['reason'] in turnover.columns:
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for ax, (name, df) in zip(axes, COHORTS.items()):
        top = df[COL['reason']].value_counts().head(10)
        top.sort_values().plot(kind='barh', ax=ax, color=COHORT_COLOUR[name],
                               edgecolor='white')
        ax.set_title(f'Top Off-Board Reasons – {name}')
        ax.set_xlabel('Number of Leavers')
        ax.set_ylabel('')
        print(f'\nTop off-board reasons – {name}:')
        print(top.to_string())
    fig.suptitle('Section 2 – Top Off-Board Reasons by Cohort', fontweight='bold', fontsize=14)
    plt.savefig('s2_top_reasons_by_cohort.png', bbox_inches='tight')
    plt.show()


# =============================================================================
# 3. VOLUNTARY VS INVOLUNTARY LEAVER PROFILE
# =============================================================================
# (Inherently comparative – contrasts the two cohorts directly.)
print('\n' + '='*65)
print('SECTION 3 – VOLUNTARY VS INVOLUNTARY LEAVER PROFILE')
print('='*65)

num_vars = [c for c in [COL['age'], COL['tenure'], COL['daily_salary'],
                         COL['base_pay'], COL['commute_km']]
            if c in turnover.columns]

# ── 3a  Summary statistics ────────────────────────────────────────────────────
profile_stats = (
    turnover.groupby(COL['category'])[num_vars]
            .agg(['mean', 'median', 'std'])
            .round(2)
)
print('\nProfile statistics by category:')
print(profile_stats.to_string())

# ── 3b  Mann-Whitney U tests ──────────────────────────────────────────────────
print('\nMann-Whitney U test (voluntary vs involuntary):')
for col in num_vars:
    grp_vol   = turnover.loc[turnover['Is_Voluntary'] == 1, col].dropna()
    grp_invol = turnover.loc[turnover['Is_Voluntary'] == 0, col].dropna()
    if len(grp_vol) > 0 and len(grp_invol) > 0:
        stat, p = mannwhitneyu(grp_vol, grp_invol, alternative='two-sided')
        sig = '***' if p < 0.001 else ('**' if p < 0.01 else ('*' if p < 0.05 else ''))
        print(f'  {col:<25}  U={stat:>8.0f}  p={p:.4f}  {sig}')

# ── 3c  Boxplots ─────────────────────────────────────────────────────────────
n_vars = len(num_vars)
fig, axes = plt.subplots(1, n_vars, figsize=(4 * n_vars, 5))
if n_vars == 1:
    axes = [axes]

for ax, col in zip(axes, num_vars):
    sns.boxplot(
        data=turnover, x=COL['category'], y=col, ax=ax,
        palette=PALETTE_2, fliersize=3, linewidth=0.8,
    )
    ax.set_title(col.replace('_', ' '))
    ax.set_xlabel('')

fig.suptitle('Section 3 – Leaver Profile Boxplots', fontweight='bold', fontsize=13)
plt.savefig('s3_profile_boxplots.png', bbox_inches='tight')
plt.show()

# ── 3d  Distribution plots ────────────────────────────────────────────────────
fig, axes = plt.subplots(1, n_vars, figsize=(4 * n_vars, 4))
if n_vars == 1:
    axes = [axes]

for ax, col in zip(axes, num_vars):
    for cat, colour in zip(turnover[COL['category']].unique(), PALETTE_2):
        subset = turnover.loc[turnover[COL['category']] == cat, col].dropna()
        sns.kdeplot(subset, ax=ax, label=cat, color=colour, fill=True, alpha=0.35)
    ax.set_title(col.replace('_', ' '))
    ax.legend(frameon=False)

fig.suptitle('Section 3 – Leaver Profile Distributions', fontweight='bold', fontsize=13)
plt.savefig('s3_profile_distributions.png', bbox_inches='tight')
plt.show()


# =============================================================================
# 4. TENURE ANALYSIS
# =============================================================================
# (Comparative – both cohorts shown side by side across tenure bands.)
print('\n' + '='*65)
print('SECTION 4 – TENURE ANALYSIS')
print('='*65)

if COL['tenure_band'] in turnover.columns:
    tenure_summary = (
        turnover.groupby([COL['tenure_band'], COL['category']])
                .size()
                .unstack(fill_value=0)
    )
    # Order rows using Tenure_Sort_Order (<1yr first, 15+yr last) rather than text
    if TENURE_BAND_ORDER:
        tenure_summary = tenure_summary.reindex(
            [b for b in TENURE_BAND_ORDER if b in tenure_summary.index]
        )

    avg_tenure_by_cat = turnover.groupby(COL['category'])[COL['tenure']].mean().round(2)
    print('\nAverage tenure by Off-Board Category:')
    print(avg_tenure_by_cat.to_string())

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Grouped bar – counts by tenure band and category
    tenure_summary.plot(kind='bar', ax=axes[0],
                        color=PALETTE_2, edgecolor='white')
    axes[0].set_title('Leavers by Tenure Band and Category')
    axes[0].set_xlabel('Tenure Band')
    axes[0].set_ylabel('Number of Leavers')
    axes[0].tick_params(axis='x', rotation=30)
    axes[0].legend(title='Category', frameon=False)

    # Share of each category within each tenure band (100% stacked)
    tenure_share = tenure_summary.div(tenure_summary.sum(axis=1), axis=0) * 100
    tenure_share.plot(kind='bar', stacked=True, ax=axes[1],
                      color=PALETTE_2, edgecolor='white')
    axes[1].set_title('Category Split within each Tenure Band (%)')
    axes[1].set_xlabel('Tenure Band')
    axes[1].set_ylabel('% of Leavers')
    axes[1].tick_params(axis='x', rotation=30)
    axes[1].yaxis.set_major_formatter(mticker.PercentFormatter())
    axes[1].legend(title='Category', frameon=False)

    fig.suptitle('Section 4 – Tenure Analysis', fontweight='bold', fontsize=13)
    plt.savefig('s4_tenure_analysis.png', bbox_inches='tight')
    plt.show()

    print('\nLeavers by Tenure Band and Category:')
    print(tenure_summary.to_string())


# =============================================================================
# 5. EARLY LEAVER ANALYSIS  (Tenure < 1 Year)  — per cohort
# =============================================================================
print('\n' + '='*65)
print('SECTION 5 – EARLY LEAVER ANALYSIS (Tenure < 1 Year AND < 2 Years)')
print('='*65)

print("""
  METHODOLOGY CAVEAT – point-in-time sampling bias
  ─────────────────────────────────────────────────────────────
  Headcount is a snapshot as at February; Turnover captures only
  employees who were present in that Feb snapshot and have since left.
  Anyone hired AFTER February who has already left is structurally
  excluded from this dataset (they were never in the Feb headcount).
  This UNDERSTATES early attrition, especially the '<1 Year' band,
  because a whole slice of "hired fast, left fast" leavers never
  enters the sample. Treat '<1 Year' / '<2 Years' figures below as a
  FLOOR on early attrition, not the true rate. Do not merge in
  post-Feb hire-and-leave records here – there is no matching
  headcount denominator for them; if needed, report that as a
  separate, clearly-labelled new-hire cohort metric instead.
""")

early_num_cols = [c for c in [COL['age'], COL['daily_salary'], COL['commute_km']]
                  if c in turnover.columns]

# Track early-leaver % per cohort per threshold for the executive summary
early_pct_by_cohort = {}   # {(cohort_name, threshold_label): pct}

def early_leaver_analysis(df, name, threshold_label, flag_col):
    """Compare early vs non-early leavers within a single cohort at one threshold."""
    if len(df) == 0 or flag_col not in df.columns:
        return
    df = df.copy()
    early     = df[df[flag_col] == 1]
    pct_early = len(early) / len(df) * 100
    early_pct_by_cohort[(name, threshold_label)] = pct_early
    print(f'\n[{name} | {threshold_label}] Early leavers: {len(early):,} '
          f'({pct_early:.1f}% of {name.lower()} exits)')

    # Boxplot: early vs non-early on key numerics
    df['Early_Label'] = df[flag_col].map({1: f'Early ({threshold_label})', 0: 'Non-Early'})
    fig, axes = plt.subplots(1, len(early_num_cols), figsize=(4 * len(early_num_cols), 5))
    if len(early_num_cols) == 1:
        axes = [axes]
    for ax, col in zip(axes, early_num_cols):
        sns.boxplot(data=df, x='Early_Label', y=col, ax=ax,
                    palette=[BRAND_ORANGE, BRAND_BLUE], fliersize=3)
        ax.set_title(col.replace('_', ' '))
        ax.set_xlabel('')
    fig.suptitle(f'Section 5 – Early vs Non-Early Leavers ({name}, {threshold_label})',
                 fontweight='bold', fontsize=13)
    plt.savefig(f's5_early_leavers_{slug(name)}_{threshold_label.strip("<> ").replace(" ", "")}.png',
                bbox_inches='tight')
    plt.show()

    # Top reasons among early leavers
    if COL['reason'] in early.columns and len(early) > 0:
        early_reasons = early[COL['reason']].value_counts().head(10)
        print(f'[{name} | {threshold_label}] Top reasons among early leavers:')
        print(early_reasons.to_string())

    # Early-leaver rate by Area and Harrods Band
    for var, label in [(COL['area'], 'Area'), (COL['harrods_band'], 'Harrods Band')]:
        if var in df.columns:
            rate = df.groupby(var)[flag_col].mean().mul(100).round(1)
            print(f'[{name} | {threshold_label}] Early leaver % by {label}:')
            print(rate.sort_values(ascending=False).head(10).to_string())

for name, df in COHORTS.items():
    for threshold_label, cutoff in EARLY_LEAVER_THRESHOLDS.items():
        early_leaver_analysis(df, name, threshold_label, f'Early_Leaver_{cutoff}Y')


# =============================================================================
# 6. COMMUTE ANALYSIS
# =============================================================================
print('\n' + '='*65)
print('SECTION 6 – COMMUTE ANALYSIS')
print('='*65)

if COL['commute_km'] in turnover.columns:
    commute_summary = (
        turnover.groupby(COL['category'])[COL['commute_km']]
                .agg(['mean', 'median', 'std'])
                .round(2)
    )
    print('\nCommute (km) summary by category:')
    print(commute_summary.to_string())

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Boxplot by category
    sns.boxplot(data=turnover, x=COL['category'], y=COL['commute_km'],
                ax=axes[0], palette=PALETTE_2, fliersize=3)
    axes[0].set_title('Commute Distance by Off-Board Category')
    axes[0].set_xlabel('')

    # Counts by commute band and category
    if COL['commute_band'] in turnover.columns:
        commute_ct = (
            turnover.groupby([COL['commute_band'], COL['category']])
                    .size().unstack(fill_value=0)
        )
        commute_ct.plot(kind='bar', ax=axes[1], color=PALETTE_2, edgecolor='white')
        axes[1].set_title('Leavers by Commute Band and Category')
        axes[1].set_xlabel('Commute Band')
        axes[1].set_ylabel('Number of Leavers')
        axes[1].tick_params(axis='x', rotation=30)
        axes[1].legend(title='Category', frameon=False)

    fig.suptitle('Section 6 – Commute Analysis', fontweight='bold', fontsize=13)
    plt.savefig('s6_commute_analysis.png', bbox_inches='tight')
    plt.show()

# ── 6b  Commute-related reasons per cohort ────────────────────────────────────
if COL['reason'] in turnover.columns:
    commute_keywords = ['commute', 'travel', 'distance', 'location', 'relocat']
    for name, df in COHORTS.items():
        hit = df[df[COL['reason']].str.lower()
                   .str.contains('|'.join(commute_keywords), na=False)]
        print(f'\n[{name}] Leavers citing commute/relocation reasons: {len(hit):,}')
        if len(hit) > 0 and COL['commute_km'] in df.columns:
            print(f'   Median commute (km): {hit[COL["commute_km"]].median():.1f}')


# =============================================================================
# 7. PAY ANALYSIS
# =============================================================================
print('\n' + '='*65)
print('SECTION 7 – PAY ANALYSIS')
print('='*65)

pay_cols = [c for c in [COL['daily_salary'], COL['base_pay']] if c in turnover.columns]

if pay_cols:
    pay_summary = (
        turnover.groupby(COL['category'])[pay_cols]
                .agg(['mean', 'median', 'std'])
                .round(2)
    )
    print('\nPay summary by Off-Board Category:')
    print(pay_summary.to_string())

    # Boxplots by category
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, col in zip(axes, pay_cols[:2]):
        sns.boxplot(data=turnover, x=COL['category'], y=col,
                    ax=ax, palette=PALETTE_2, fliersize=3)
        ax.set_title(f'{col.replace("_", " ")} by Category')
        ax.set_xlabel('')
    fig.suptitle('Section 7 – Pay Analysis', fontweight='bold', fontsize=13)
    plt.savefig('s7_pay_boxplots.png', bbox_inches='tight')
    plt.show()

    # Median daily salary by reason, per cohort (vol & invol shown separately)
    if COL['reason'] in turnover.columns and COL['daily_salary'] in turnover.columns:
        for name in ['Voluntary', 'Involuntary']:
            df = COHORTS[name]
            if len(df) == 0:
                continue
            pay_by_reason = df.groupby(COL['reason'])[COL['daily_salary']].median().sort_values()
            fig, ax = plt.subplots(figsize=(10, max(4, len(pay_by_reason) * 0.4)))
            pay_by_reason.plot(kind='barh', ax=ax, color=COHORT_COLOUR[name])
            ax.set_title(f'Median Daily Salary by Off-Board Reason ({name})')
            ax.set_xlabel('Median Daily Salary (£)')
            plt.savefig(f's7_pay_by_reason_{slug(name)}.png', bbox_inches='tight')
            plt.show()

    # Pay-dissatisfaction comparison (voluntary leavers only – reason is a resignation)
    if COL['reason'] in vol.columns and COL['daily_salary'] in vol.columns:
        pay_keywords = ['pay', 'salary', 'compensation', 'wage', 'remuneration']
        pay_leavers     = vol[vol[COL['reason']].str.lower()
                               .str.contains('|'.join(pay_keywords), na=False)]
        non_pay_leavers = vol[~vol[COL['reason']].str.lower()
                               .str.contains('|'.join(pay_keywords), na=False)]
        print(f'\n  Pay-related voluntary leavers : {len(pay_leavers):,}')
        if len(pay_leavers) > 0:
            print(f'  Median salary (pay leavers)   : £{pay_leavers[COL["daily_salary"]].median():,.0f}')
            print(f'  Median salary (other vol)     : £{non_pay_leavers[COL["daily_salary"]].median():,.0f}')

    # Pay by Harrods Band and category
    if COL['harrods_band'] in turnover.columns and COL['daily_salary'] in turnover.columns:
        pay_band = (
            turnover.groupby([COL['harrods_band'], COL['category']])[COL['daily_salary']]
                    .median().unstack()
        )
        fig, ax = plt.subplots(figsize=(10, 5))
        pay_band.plot(kind='bar', ax=ax, color=PALETTE_2, edgecolor='white')
        ax.set_title('Median Daily Salary by Harrods Band and Category')
        ax.set_xlabel('Harrods Band')
        ax.set_ylabel('Median Daily Salary (£)')
        ax.tick_params(axis='x', rotation=30)
        ax.legend(title='Category', frameon=False)
        plt.savefig('s7_pay_by_band.png', bbox_inches='tight')
        plt.show()


# =============================================================================
# 8. DEPARTMENT & ORGANISATIONAL HOTSPOT ANALYSIS  — per cohort
# =============================================================================
print('\n' + '='*65)
print('SECTION 8 – ORGANISATIONAL HOTSPOT ANALYSIS')
print('='*65)

org_vars = [c for c in [COL['area'], COL['division'], COL['business_unit'],
                         COL['harrods_band'], COL['job_profile']]
            if c in turnover.columns]

# Track top hotspots per cohort for the executive summary
hotspots = {}

def hotspot_analysis(df, name):
    """Ranked tables + bar charts of leaver counts by org dimension for a cohort."""
    if len(df) == 0:
        return
    hotspots[name] = {}
    for var in org_vars:
        counts   = df[var].value_counts()
        share    = (counts / counts.sum() * 100).round(1)
        # Rate = this cohort's leavers as a share of ALL leavers in that group
        all_by   = turnover[var].value_counts()
        pct_all  = (counts / all_by * 100).round(1)
        ranked = pd.DataFrame({
            'Leavers'          : counts,
            'Pct_of_Cohort'    : share,
            'Pct_of_All_Leavers': pct_all,
        }).sort_values('Leavers', ascending=False).head(15)
        print(f'\n[{name}] Top hotspots by {var}:')
        print(ranked.to_string())
        if var in (COL['area'], COL['division']):
            hotspots[name][var] = counts.idxmax() if len(counts) else 'N/A'

    # Bar charts for the first three org dimensions
    show_vars = org_vars[:3]
    fig, axes = plt.subplots(1, len(show_vars), figsize=(6 * len(show_vars), 5))
    if len(show_vars) == 1:
        axes = [axes]
    for ax, var in zip(axes, show_vars):
        top = df[var].value_counts().head(10)
        top.sort_values().plot(kind='barh', ax=ax, color=COHORT_COLOUR[name])
        ax.set_title(f'{var.replace("_", " ")}')
        ax.set_xlabel('Leavers')
    fig.suptitle(f'Section 8 – Organisational Hotspots ({name})',
                 fontweight='bold', fontsize=13)
    plt.savefig(f's8_hotspots_{slug(name)}.png', bbox_inches='tight')
    plt.show()

    # Area x Division heatmap
    if COL['area'] in df.columns and COL['division'] in df.columns:
        heat = df.groupby([COL['area'], COL['division']]).size().unstack(fill_value=0)
        fig, ax = plt.subplots(figsize=(max(10, heat.shape[1] * 1.2),
                                        max(6, heat.shape[0] * 0.6)))
        sns.heatmap(heat, annot=True, fmt='d', cmap='YlOrRd',
                    linewidths=0.5, ax=ax, cbar_kws={'label': 'Leavers'})
        ax.set_title(f'Section 8 – Leavers Heatmap: Area × Division ({name})')
        ax.set_xlabel('Division')
        ax.set_ylabel('Area')
        plt.savefig(f's8_heatmap_area_division_{slug(name)}.png', bbox_inches='tight')
        plt.show()

for name, df in COHORTS.items():
    hotspot_analysis(df, name)


# =============================================================================
# 9. OFF-BOARD REASON ANALYSIS  — voluntary and involuntary
# =============================================================================
print('\n' + '='*65)
print('SECTION 9 – OFF-BOARD REASON ANALYSIS')
print('='*65)

def reason_analysis(df, name):
    """Cross-tabulate off-board reasons against key segments for a cohort."""
    if len(df) == 0 or COL['reason'] not in df.columns:
        return
    top_reasons = df[COL['reason']].value_counts()
    print(f'\n[{name}] Top off-board reasons:')
    print(top_reasons.head(15).to_string())

    breakdown_vars = [c for c in [COL['tenure_band'], COL['age_band'],
                                   COL['harrods_band'], COL['area']]
                      if c in df.columns]

    for var in breakdown_vars:
        ct = pd.crosstab(df[var], df[COL['reason']])
        if ct.empty:
            continue

        # Heatmap
        fig, ax = plt.subplots(figsize=(max(12, ct.shape[1] * 1.0),
                                        max(5, ct.shape[0] * 0.5)))
        sns.heatmap(ct, annot=True, fmt='d', cmap='Blues',
                    linewidths=0.3, ax=ax, cbar_kws={'label': 'Leavers'})
        ax.set_title(f'Off-Board Reason by {var.replace("_", " ")} ({name})')
        ax.set_xlabel('Off-Board Reason')
        ax.set_ylabel(var.replace('_', ' '))
        plt.savefig(f's9_heatmap_{var}_{slug(name)}.png', bbox_inches='tight')
        plt.show()

        # 100% stacked bar of top reasons
        ct_pct = ct.div(ct.sum(axis=1), axis=0) * 100
        top_list = top_reasons.head(8).index.tolist()
        ct_pct_top = ct_pct[[c for c in top_list if c in ct_pct.columns]]
        fig, ax = plt.subplots(figsize=(10, 5))
        ct_pct_top.plot(kind='bar', stacked=True, ax=ax, colormap='tab10', edgecolor='white')
        ax.set_title(f'Reason Breakdown by {var.replace("_", " ")} – {name} (%)')
        ax.set_xlabel(var.replace('_', ' '))
        ax.set_ylabel('% of Leavers')
        ax.legend(title='Reason', bbox_to_anchor=(1.05, 1), loc='upper left', frameon=False)
        ax.tick_params(axis='x', rotation=30)
        plt.savefig(f's9_stackedbar_{var}_{slug(name)}.png', bbox_inches='tight')
        plt.show()

# Reasons are most meaningful within each category, so run for Voluntary & Involuntary
for name in ['Voluntary', 'Involuntary']:
    reason_analysis(COHORTS[name], name)


# =============================================================================
# 10. ATTRITION PERSONAS  — per cohort
# =============================================================================
print('\n' + '='*65)
print('SECTION 10 – ATTRITION PERSONAS')
print('='*65)

# Voluntary-oriented persona themes
PERSONA_MAP_VOL = {
    'Career':             ['Development Opportunities', 'Hired by Competitor', 'Return to Education'],
    'Employee Experience':['Culture', 'Working Environment'],
    'Lifestyle':          ['Commute Time', 'Relocation', 'Work Life Balance'],
    'Personal':           ['Family Reasons', 'Ill Health', 'Visa Expiry'],
}
# Involuntary-oriented persona themes
PERSONA_MAP_INVOL = {
    'Performance':   ['Performance', 'Capability', 'Conduct', 'Dismissal', 'Gross Misconduct'],
    'Restructure':   ['Redundancy', 'Restructure', 'Reorganis'],
    'Contractual':   ['End Of Contract', 'Fixed Term', 'Probation', 'Contract'],
    'Other Exit':    ['Retirement', 'Tupe', 'Death'],
}

def build_persona(reason, mapping):
    if pd.isna(reason):
        return 'Other'
    r = str(reason).lower()
    for persona, keywords in mapping.items():
        for kw in keywords:
            if kw.lower() in r:
                return persona
    return 'Other'

def persona_analysis(df, name, mapping):
    if len(df) == 0 or COL['reason'] not in df.columns:
        return
    df = df.copy()
    df['Persona'] = df[COL['reason']].apply(lambda x: build_persona(x, mapping))

    persona_counts = df['Persona'].value_counts()
    print(f'\n[{name}] Persona distribution:')
    print(persona_counts.to_string())

    profile_cols = [c for c in [COL['age'], COL['tenure'],
                                  COL['daily_salary'], COL['commute_km']]
                    if c in df.columns]
    if profile_cols:
        print(f'[{name}] Persona profiles (mean | median):')
        print(df.groupby('Persona')[profile_cols].agg(['mean', 'median']).round(2).to_string())

        fig, axes = plt.subplots(1, len(profile_cols), figsize=(4 * len(profile_cols), 5))
        if len(profile_cols) == 1:
            axes = [axes]
        palette_p = sns.color_palette('Set2', n_colors=df['Persona'].nunique())
        for ax, col in zip(axes, profile_cols):
            sns.violinplot(data=df, x='Persona', y=col, ax=ax,
                           palette=palette_p, inner='box', linewidth=0.8)
            ax.set_title(col.replace('_', ' '))
            ax.set_xlabel('')
            ax.tick_params(axis='x', rotation=30)
        fig.suptitle(f'Section 10 – Persona Profiles ({name})', fontweight='bold', fontsize=13)
        plt.savefig(f's10_persona_profiles_{slug(name)}.png', bbox_inches='tight')
        plt.show()

    # Donut of persona share
    fig, ax = plt.subplots(figsize=(7, 7))
    wedges, texts, autotexts = ax.pie(
        persona_counts.values, labels=persona_counts.index, autopct='%1.1f%%',
        startangle=90, wedgeprops={'width': 0.55, 'edgecolor': 'white', 'linewidth': 2},
        colors=sns.color_palette('Set2', n_colors=len(persona_counts)),
    )
    for at in autotexts:
        at.set_fontweight('bold')
    ax.set_title(f'Section 10 – Attrition Personas ({name})', fontweight='bold')
    plt.savefig(f's10_persona_donut_{slug(name)}.png', bbox_inches='tight')
    plt.show()

persona_analysis(COHORTS['Voluntary'],   'Voluntary',   PERSONA_MAP_VOL)
persona_analysis(COHORTS['Involuntary'], 'Involuntary', PERSONA_MAP_INVOL)


# =============================================================================
# 11. STATISTICAL DRIVER ANALYSIS (Chi-Square + Cramér's V)
# =============================================================================
# Tests association of each categorical variable with the voluntary vs
# involuntary split – inherently a "both cohorts" analysis.
print('\n' + '='*65)
print('SECTION 11 – STATISTICAL DRIVER ANALYSIS')
print('='*65)

def cramers_v(confusion_matrix):
    chi2 = chi2_contingency(confusion_matrix)[0]
    n    = confusion_matrix.sum().sum()
    phi2 = chi2 / n
    r, k = confusion_matrix.shape
    return np.sqrt(phi2 / (min(k - 1, r - 1))) if min(k - 1, r - 1) > 0 else 0.0

cat_test_vars = [c for c in [COL['tenure_band'], COL['harrods_band'],
                               COL['area'], COL['division'], COL['commute_band']]
                 if c in turnover.columns]

chi_results = []
for var in cat_test_vars:
    ct = pd.crosstab(turnover[var], turnover['Is_Voluntary'])
    chi2, p, dof, _ = chi2_contingency(ct)
    cv = cramers_v(ct)
    chi_results.append({'Variable': var, 'Chi2': round(chi2, 2),
                        'p_value': round(p, 4), 'Cramers_V': round(cv, 3),
                        'Significant': 'Yes' if p < 0.05 else 'No'})

chi_df = pd.DataFrame(chi_results).sort_values('Cramers_V', ascending=False)
print('\nChi-Square Driver Analysis (Voluntary vs Involuntary):')
print(chi_df.to_string(index=False))

fig, ax = plt.subplots(figsize=(8, 4))
ax.barh(chi_df['Variable'], chi_df['Cramers_V'],
        color=[BRAND_BLUE if s == 'Yes' else BRAND_ORANGE for s in chi_df['Significant']])
ax.axvline(0.1, color='gray', linestyle='--', linewidth=0.8, label='Small effect (0.1)')
ax.axvline(0.3, color='gray', linestyle=':',  linewidth=0.8, label='Medium effect (0.3)')
ax.set_title("Section 11 – Cramér's V Effect Size by Variable")
ax.set_xlabel("Cramér's V")
ax.legend(frameon=False, fontsize=8)
plt.savefig('s11_chi_square_effect_sizes.png', bbox_inches='tight')
plt.show()


# =============================================================================
# 12. LOGISTIC REGRESSION  (target: Is_Voluntary → contrasts both cohorts)
# =============================================================================
print('\n' + '='*65)
print('SECTION 12 – LOGISTIC REGRESSION')
print('='*65)

lr_num_cols = [c for c in [COL['age'], COL['tenure'],
                             COL['daily_salary'], COL['commute_km']]
               if c in turnover.columns]
lr_cat_cols = [c for c in [COL['harrods_band']] if c in turnover.columns]

lr_df = turnover[lr_num_cols + lr_cat_cols + ['Is_Voluntary']].dropna()
if lr_cat_cols:
    lr_df = pd.get_dummies(lr_df, columns=lr_cat_cols, drop_first=True)

feature_cols = [c for c in lr_df.columns if c != 'Is_Voluntary']
X = lr_df[feature_cols].astype(float)
y = lr_df['Is_Voluntary']

# VIF
X_vif = sm.add_constant(X)
vif_data = pd.DataFrame({
    'Feature': X_vif.columns,
    'VIF': [variance_inflation_factor(X_vif.values, i) for i in range(X_vif.shape[1])]
}).sort_values('VIF', ascending=False)
print('\nVariance Inflation Factors:')
print(vif_data.to_string(index=False))

# Fit
X_const = sm.add_constant(X)
model = sm.Logit(y, X_const).fit(disp=False)
print('\nLogistic Regression Summary (target = Is_Voluntary):')
print(model.summary2())

# Odds ratios
odds_df = pd.DataFrame({
    'Feature'   : model.params.index,
    'Coef'      : model.params.values.round(4),
    'Odds_Ratio': np.exp(model.params.values).round(4),
    'p_value'   : model.pvalues.values.round(4),
    'CI_Lower'  : np.exp(model.conf_int()[0].values).round(4),
    'CI_Upper'  : np.exp(model.conf_int()[1].values).round(4),
}).sort_values('Odds_Ratio', ascending=False)
print('\nOdds Ratio Interpretation Table (OR>1 => more likely VOLUNTARY):')
print(odds_df.to_string(index=False))

# Forest plot
odds_plot = odds_df[odds_df['Feature'] != 'const'].copy()
fig, ax = plt.subplots(figsize=(8, max(4, len(odds_plot) * 0.5)))
y_pos = range(len(odds_plot))
ax.barh(y_pos, odds_plot['Odds_Ratio'], xerr=[
    odds_plot['Odds_Ratio'] - odds_plot['CI_Lower'],
    odds_plot['CI_Upper']   - odds_plot['Odds_Ratio'],
], color=BRAND_BLUE, alpha=0.7, capsize=3)
ax.axvline(1, color='black', linewidth=0.8, linestyle='--')
ax.set_yticks(list(y_pos))
ax.set_yticklabels(odds_plot['Feature'].str.replace('_', ' '))
ax.set_title('Section 12 – Logistic Regression: Odds Ratios')
ax.set_xlabel('Odds Ratio (ref = 1)')
plt.savefig('s12_odds_ratios.png', bbox_inches='tight')
plt.show()


# =============================================================================
# 13. FEATURE IMPACT ON ATTRITION  (univariate + multivariate)
# =============================================================================
# Goal: which individual features are most associated with an employee being a
# VOLUNTARY leaver (univariate), and which COMBINATIONS of features carry the
# highest risk (multivariate) — not how features correlate with each other.
print('\n' + '='*65)
print('SECTION 13 – FEATURE IMPACT ON ATTRITION (Univariate + Multivariate)')
print('='*65)

from scipy.stats import pointbiserialr

# ── 13a  Univariate – numeric features vs Is_Voluntary ──────────────────────
# Point-biserial correlation = Pearson r between a continuous variable and a
# binary target. It answers: "as this number goes up, does the odds of
# leaving VOLUNTARILY (vs involuntarily) go up or down?"
impact_num_cols = [c for c in [COL['age'], COL['tenure'], COL['daily_salary'],
                                 COL['base_pay'], COL['commute_km']]
                   if c in turnover.columns]

pb_results = []
for col in impact_num_cols:
    valid = turnover[[col, 'Is_Voluntary']].dropna()
    if len(valid) > 2:
        r, p = pointbiserialr(valid['Is_Voluntary'], valid[col])
        pb_results.append({'Feature': col, 'Point_Biserial_r': round(r, 3),
                           'p_value': round(p, 4),
                           'Significant': 'Yes' if p < 0.05 else 'No'})

pb_df = pd.DataFrame(pb_results).sort_values('Point_Biserial_r', key=abs, ascending=False)
print('\n[Univariate – Numeric] Correlation of each feature with being a VOLUNTARY leaver:')
print(pb_df.to_string(index=False))
print('  (positive r => higher values associated with VOLUNTARY exit;')
print('   negative r => higher values associated with INVOLUNTARY exit)')

fig, ax = plt.subplots(figsize=(8, 4))
colours = [BRAND_BLUE if s == 'Yes' else BRAND_ORANGE for s in pb_df['Significant']]
ax.barh(pb_df['Feature'].str.replace('_', ' '), pb_df['Point_Biserial_r'], color=colours)
ax.axvline(0, color='black', linewidth=0.8)
ax.set_title('Section 13a – Numeric Feature Impact on Voluntary Attrition')
ax.set_xlabel('Point-Biserial Correlation with Is_Voluntary')
plt.savefig('s13a_numeric_feature_impact.png', bbox_inches='tight')
plt.show()

# ── 13b  Univariate – categorical features vs Is_Voluntary ──────────────────
# Reuses the Cramér's V effect sizes already computed in Section 11 (each
# tests association between a categorical feature and Is_Voluntary).
print("\n[Univariate – Categorical] Cramér's V association with voluntary/involuntary split:")
print(chi_df.to_string(index=False))

# ── 13c  Combined univariate ranking (numeric + categorical) ────────────────
combined_impact = pd.concat([
    pb_df.assign(Type='Numeric', Effect_Size=pb_df['Point_Biserial_r'].abs())[
        ['Feature', 'Type', 'Effect_Size', 'p_value', 'Significant']],
    chi_df.assign(Type='Categorical', Effect_Size=chi_df['Cramers_V'])[
        ['Variable', 'Type', 'Effect_Size', 'p_value', 'Significant']
    ].rename(columns={'Variable': 'Feature'}),
], ignore_index=True).sort_values('Effect_Size', ascending=False)

print('\n[Combined Univariate Ranking] All features ranked by effect size'
      ' (|r| for numeric, Cramér\'s V for categorical):')
print(combined_impact.to_string(index=False))

fig, ax = plt.subplots(figsize=(9, max(4, len(combined_impact) * 0.45)))
colours = [BRAND_BLUE if t == 'Numeric' else BRAND_ORANGE for t in combined_impact['Type']]
ax.barh(combined_impact['Feature'].str.replace('_', ' '), combined_impact['Effect_Size'], color=colours)
ax.invert_yaxis()
ax.set_title('Section 13b – Combined Univariate Feature Ranking (Impact on Attrition Type)')
ax.set_xlabel('Effect Size (|point-biserial r| or Cramér\'s V)')
from matplotlib.patches import Patch
ax.legend(handles=[Patch(color=BRAND_BLUE, label='Numeric'),
                    Patch(color=BRAND_ORANGE, label='Categorical')], frameon=False)
plt.savefig('s13b_combined_univariate_ranking.png', bbox_inches='tight')
plt.show()

# ── 13d  Multivariate – which COMBINATIONS of features drive attrition ──────
# A single linear model (Section 12) can miss non-linear interactions
# (e.g. "young AND long commute" being riskier than either factor alone).
# A Random Forest captures these combinations and ranks feature importance
# in a way that reflects interaction effects, not just individual linear
# association.
try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.inspection import permutation_importance

    rf_cat_cols = [c for c in [COL['harrods_band'], COL['tenure_band'],
                                 COL['area'], COL['commute_band']]
                   if c in turnover.columns]
    rf_df = turnover[impact_num_cols + rf_cat_cols + ['Is_Voluntary']].dropna()
    rf_df_enc = pd.get_dummies(rf_df, columns=rf_cat_cols, drop_first=True)

    rf_features = [c for c in rf_df_enc.columns if c != 'Is_Voluntary']
    X_rf = rf_df_enc[rf_features]
    y_rf = rf_df_enc['Is_Voluntary']

    rf_model = RandomForestClassifier(
        n_estimators=300, max_depth=5, random_state=42, class_weight='balanced'
    )
    rf_model.fit(X_rf, y_rf)

    # Built-in importance captures interactions the tree structure exploits
    rf_importance = pd.Series(rf_model.feature_importances_, index=rf_features)

    # Permutation importance – more robust, measures accuracy drop when a
    # feature is shuffled, so correlated/combined effects are reflected too
    perm = permutation_importance(rf_model, X_rf, y_rf, n_repeats=20, random_state=42)
    perm_importance = pd.Series(perm.importances_mean, index=rf_features)

    rf_summary = pd.DataFrame({
        'Feature': rf_features,
        'RF_Importance': rf_importance.values.round(4),
        'Permutation_Importance': perm_importance.values.round(4),
    }).sort_values('Permutation_Importance', ascending=False)

    print('\n[Multivariate – Random Forest] Feature importance '
          '(captures non-linear combinations/interactions):')
    print(rf_summary.head(15).to_string(index=False))

    fig, ax = plt.subplots(figsize=(9, max(4, min(15, len(rf_summary)) * 0.4)))
    top_rf = rf_summary.head(15).iloc[::-1]
    ax.barh(top_rf['Feature'].str.replace('_', ' '), top_rf['Permutation_Importance'], color=BRAND_GREEN)
    ax.set_title('Section 13c – Multivariate Feature Importance (Random Forest)')
    ax.set_xlabel('Permutation Importance (accuracy drop when shuffled)')
    plt.savefig('s13c_multivariate_feature_importance.png', bbox_inches='tight')
    plt.show()

except ImportError:
    print('\n[Multivariate] scikit-learn not installed – skipping Random Forest '
          'importance. Run: pip install scikit-learn')
    rf_summary = pd.DataFrame()

# ── 13e  Highest-risk feature COMBINATIONS (interaction heatmap) ────────────
# Takes the top-2 categorical drivers from the Cramér's V ranking and shows
# voluntary-exit share for every combination of their categories, to surface
# specific high-risk segments (e.g. "Area X + Tenure Band <1yr").
top2_cat = chi_df.sort_values('Cramers_V', ascending=False)['Variable'].head(2).tolist()
if len(top2_cat) == 2:
    var1, var2 = top2_cat
    combo_ct = pd.crosstab(turnover[var1], turnover[var2])
    combo_vol_pct = (
        turnover.groupby([var1, var2])['Is_Voluntary'].mean().mul(100).unstack()
    )
    combo_n = turnover.groupby([var1, var2]).size().unstack(fill_value=0)
    # Mask combinations with too few leavers to be meaningful (n < 5)
    combo_vol_pct_masked = combo_vol_pct.where(combo_n >= 5)

    fig, ax = plt.subplots(figsize=(max(10, combo_vol_pct_masked.shape[1] * 1.1),
                                    max(6, combo_vol_pct_masked.shape[0] * 0.6)))
    sns.heatmap(combo_vol_pct_masked, annot=True, fmt='.0f', cmap='RdYlGn_r',
                linewidths=0.5, ax=ax, cbar_kws={'label': '% Voluntary'})
    ax.set_title(f'Section 13d – Voluntary Exit % by {var1.replace("_"," ")} '
                 f'× {var2.replace("_"," ")} (cells with n<5 hidden)')
    ax.set_xlabel(var2.replace('_', ' '))
    ax.set_ylabel(var1.replace('_', ' '))
    plt.savefig('s13d_top_risk_combination_heatmap.png', bbox_inches='tight')
    plt.show()

    print(f'\n[Multivariate – Interaction] Voluntary exit % by {var1} × {var2} '
          f'(cells with fewer than 5 leavers excluded as unreliable):')
    print(combo_vol_pct_masked.round(1).to_string())

print("""
Interpretation note:
  Point-biserial r and Cramér's V show ASSOCIATION between a feature and
  whether an exit was voluntary vs involuntary — not causation. Random
  Forest / permutation importance captures non-linear effects and feature
  combinations that a single correlation coefficient cannot, but importance
  scores still describe predictive association within this dataset, not a
  guaranteed causal driver. Use these rankings to prioritise where to
  investigate further (e.g. stay interviews, pay benchmarking) rather than
  as proof of what causes attrition.
""")


# =============================================================================
# 14. EXECUTIVE SUMMARY OUTPUT  (covers both cohorts)
# =============================================================================
print('\n' + '='*65)
print('SECTION 14 – EXECUTIVE SUMMARY')
print('='*65)

def top_n_reasons(df, n=3):
    if COL['reason'] in df.columns and len(df):
        return df[COL['reason']].value_counts().head(n).index.tolist()
    return ['N/A']

vol_reasons   = top_n_reasons(vol)
invol_reasons = top_n_reasons(invol)

vol_area   = hotspots.get('Voluntary', {}).get(COL['area'], 'N/A')
invol_area = hotspots.get('Involuntary', {}).get(COL['area'], 'N/A')
vol_div    = hotspots.get('Voluntary', {}).get(COL['division'], 'N/A')
invol_div  = hotspots.get('Involuntary', {}).get(COL['division'], 'N/A')

vol_early_pct_1y   = early_pct_by_cohort.get(('Voluntary', '<1 Year'), float('nan'))
vol_early_pct_2y   = early_pct_by_cohort.get(('Voluntary', '<2 Years'), float('nan'))
invol_early_pct_1y = early_pct_by_cohort.get(('Involuntary', '<1 Year'), float('nan'))
invol_early_pct_2y = early_pct_by_cohort.get(('Involuntary', '<2 Years'), float('nan'))

significant_drivers = chi_df[chi_df['Significant'] == 'Yes']['Variable'].tolist()
top_sig_driver = significant_drivers[0] if significant_drivers else 'N/A'
try:
    top_lr_predictor = odds_plot.iloc[0]['Feature'].replace('_', ' ')
except Exception:
    top_lr_predictor = 'N/A'

try:
    top_univariate_feature = combined_impact.iloc[0]['Feature'].replace('_', ' ')
except Exception:
    top_univariate_feature = 'N/A'

try:
    top_multivariate_feature = rf_summary.iloc[0]['Feature'].replace('_', ' ') if not rf_summary.empty else 'N/A'
except Exception:
    top_multivariate_feature = 'N/A'

print(f"""
╔══════════════════════════════════════════════════════════════╗
║            EMPLOYEE ATTRITION – EXECUTIVE SUMMARY            ║
╚══════════════════════════════════════════════════════════════╝

  HEADLINE METRICS
  ─────────────────────────────────────────────────────────────
  • Total leavers         : {total:,}
  • Voluntary attrition   : {total_vol:,}  ({pct_vol:.1f}%)
  • Involuntary attrition : {total_invol:,}  ({pct_invol:.1f}%)
  • Early exits (<1yr)    : Voluntary {vol_early_pct_1y:.1f}% | Involuntary {invol_early_pct_1y:.1f}%
  • Early exits (<2yrs)   : Voluntary {vol_early_pct_2y:.1f}% | Involuntary {invol_early_pct_2y:.1f}%
    (NOTE: these are a floor, not a true rate — see Section 5 caveat on
     point-in-time sampling bias re: post-Feb hires who already left)

  TOP INSIGHTS – VOLUNTARY
  ─────────────────────────────────────────────────────────────
  • Top reasons  : {', '.join(vol_reasons)}
  • Top Area     : {vol_area}
  • Top Division : {vol_div}
  • {vol_early_pct_1y:.1f}% of voluntary exits occur within the first year;
    {vol_early_pct_2y:.1f}% within the first two years.

  TOP INSIGHTS – INVOLUNTARY
  ─────────────────────────────────────────────────────────────
  • Top reasons  : {', '.join(invol_reasons)}
  • Top Area     : {invol_area}
  • Top Division : {invol_div}
  • {invol_early_pct_1y:.1f}% of involuntary exits occur within the first year;
    {invol_early_pct_2y:.1f}% within the first two years.

  SIGNIFICANT STATISTICAL FINDINGS (Voluntary vs Involuntary)
  ─────────────────────────────────────────────────────────────
  • Variables significantly associated with the exit type
    (Chi-Square, p < 0.05):
    {', '.join(significant_drivers) if significant_drivers else 'None significant'}
  • Strongest categorical driver (Cramér's V): {top_sig_driver}
  • Strongest regression predictor of voluntary exit: {top_lr_predictor}
  • Top univariate feature overall (Section 13b): {top_univariate_feature}
  • Top multivariate feature, incl. interactions (Section 13c): {top_multivariate_feature}

  POTENTIAL RETENTION / WORKFORCE ACTIONS
  ─────────────────────────────────────────────────────────────
  1. Strengthen onboarding & 90-day check-ins to reduce early
     voluntary exits.
  2. Review pay competitiveness where pay-related resignations
     cluster.
  3. Offer hybrid/remote options for long-commute employees
     citing commute or relocation.
  4. Invest in career pathways for high-risk tenure bands.
  5. Run stay interviews in voluntary hotspot Areas/Divisions.
  6. For involuntary exits, review recruitment quality and
     probation management in the hotspot areas identified above.

══════════════════════════════════════════════════════════════
  All charts saved as PNG files in the working directory.
  Cohorts analysed: All Leavers, Voluntary, Involuntary.
  Analysis complete.
══════════════════════════════════════════════════════════════
""")
