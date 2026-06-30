# =============================================================================
# EMPLOYEE ATTRITION ANALYSIS
# File: attrition_analysis.py
# Description: Exploratory and diagnostic attrition analysis identifying
#              patterns and potential drivers of voluntary attrition.
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
# Map cleaned names → canonical names used throughout the script.
# Adjust the left-hand keys if your file uses slightly different names.
COL = {
    'category'         : 'Offboard_Category',
    'reason'           : 'Offboard_Reason',
    'age'              : 'Age',
    'age_band'         : 'Age_Band',
    'tenure'           : 'Tenure_Years',
    'tenure_band'      : 'Tenure_Band',
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
    turnover['Is_Voluntary'] = (
        turnover[COL['category']].str.strip().str.title() == 'Voluntary'
    ).astype(int)
else:
    raise KeyError(f"Column '{COL['category']}' not found – check sheet/column names.")

# ── 1.7  Early-leaver flag ───────────────────────────────────────────────────
if COL['tenure'] in turnover.columns:
    turnover['Early_Leaver'] = (turnover[COL['tenure']] < 1).astype(int)

# ── 1.8  Voluntary-only subset ───────────────────────────────────────────────
vol = turnover[turnover['Is_Voluntary'] == 1].copy()

print('\nData preparation complete.')
print(f'  Total records  : {len(turnover):,}')
print(f'  Voluntary      : {turnover["Is_Voluntary"].sum():,}')
print(f'  Involuntary    : {(turnover["Is_Voluntary"] == 0).sum():,}')


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

# ── 2a  Donut chart – vol vs invol ───────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(16, 5))

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

# ── 2b  Monthly attrition trend (derived from Termination_Date) ──────────────
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

# ── 2c  Top off-board reasons ────────────────────────────────────────────────
if COL['reason'] in vol.columns:
    top_reasons = vol[COL['reason']].value_counts().head(12)
    top_reasons.sort_values().plot(
        kind='barh', ax=axes[2], color=BRAND_BLUE, edgecolor='white'
    )
    axes[2].set_title('Top Voluntary Off-Board Reasons')
    axes[2].set_xlabel('Number of Leavers')
    axes[2].set_ylabel('')

fig.suptitle('Section 2 – High-Level Attrition Summary', fontweight='bold', fontsize=14)
plt.savefig('s2_high_level_summary.png', bbox_inches='tight')
plt.show()

print('\nTop 12 voluntary off-board reasons:')
print(top_reasons.to_string())


# =============================================================================
# 3. VOLUNTARY VS INVOLUNTARY LEAVER PROFILE
# =============================================================================
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
print('\n' + '='*65)
print('SECTION 4 – TENURE ANALYSIS')
print('='*65)

TENURE_ORDER = ['<1 Year', '1-2 Years', '2-3 Years', '3-5 Years', '5+ Years']

if COL['tenure_band'] in turnover.columns:
    # Reorder if categories match
    existing_bands = turnover[COL['tenure_band']].dropna().unique().tolist()
    order = [b for b in TENURE_ORDER if b in existing_bands] + \
            [b for b in existing_bands if b not in TENURE_ORDER]

    tenure_summary = (
        turnover.groupby([COL['tenure_band'], COL['category']])
                .size()
                .unstack(fill_value=0)
    )
    if order:
        tenure_summary = tenure_summary.reindex(
            [o for o in order if o in tenure_summary.index]
        )

    avg_tenure_by_cat = turnover.groupby(COL['category'])[COL['tenure']].mean().round(2)
    print('\nAverage tenure by Off-Board Category:')
    print(avg_tenure_by_cat.to_string())

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Stacked bar
    tenure_summary.plot(kind='bar', stacked=True, ax=axes[0],
                        color=PALETTE_2, edgecolor='white')
    axes[0].set_title('Leavers by Tenure Band')
    axes[0].set_xlabel('Tenure Band')
    axes[0].set_ylabel('Number of Leavers')
    axes[0].tick_params(axis='x', rotation=30)
    axes[0].legend(title='Category', frameon=False)

    # Voluntary % by tenure band
    vol_pct_tenure = (
        turnover.groupby(COL['tenure_band'])['Is_Voluntary']
                .mean()
                .mul(100)
                .reindex([o for o in order if o in turnover[COL['tenure_band']].unique()])
    )
    axes[1].bar(range(len(vol_pct_tenure)), vol_pct_tenure.values, color=BRAND_BLUE)
    axes[1].set_xticks(range(len(vol_pct_tenure)))
    axes[1].set_xticklabels(vol_pct_tenure.index, rotation=30, ha='right')
    axes[1].set_title('Voluntary Attrition % by Tenure Band')
    axes[1].set_ylabel('% Voluntary')
    axes[1].yaxis.set_major_formatter(mticker.PercentFormatter())

    fig.suptitle('Section 4 – Tenure Analysis', fontweight='bold', fontsize=13)
    plt.savefig('s4_tenure_analysis.png', bbox_inches='tight')
    plt.show()

    print('\nVoluntary % by Tenure Band:')
    print(vol_pct_tenure.round(1).to_string())


# =============================================================================
# 5. EARLY LEAVER ANALYSIS
# =============================================================================
print('\n' + '='*65)
print('SECTION 5 – EARLY LEAVER ANALYSIS (Tenure < 1 Year)')
print('='*65)

early_vol = vol[vol['Early_Leaver'] == 1].copy()
non_early_vol = vol[vol['Early_Leaver'] == 0].copy()

pct_early = len(early_vol) / len(vol) * 100 if len(vol) > 0 else 0
print(f'\n  Voluntary early leavers  : {len(early_vol):,}  ({pct_early:.1f}% of voluntary exits)')

# ── 5a  Numeric comparison ───────────────────────────────────────────────────
early_num_cols = [c for c in [COL['age'], COL['daily_salary'], COL['commute_km']]
                  if c in vol.columns]

fig, axes = plt.subplots(1, len(early_num_cols), figsize=(4 * len(early_num_cols), 5))
if len(early_num_cols) == 1:
    axes = [axes]

vol['Early_Label'] = vol['Early_Leaver'].map({1: 'Early (<1yr)', 0: 'Non-Early'})
for ax, col in zip(axes, early_num_cols):
    sns.boxplot(data=vol, x='Early_Label', y=col, ax=ax,
                palette=[BRAND_ORANGE, BRAND_BLUE], fliersize=3)
    ax.set_title(col.replace('_', ' '))
    ax.set_xlabel('')

fig.suptitle('Section 5 – Early vs Non-Early Voluntary Leavers', fontweight='bold', fontsize=13)
plt.savefig('s5_early_leavers.png', bbox_inches='tight')
plt.show()

# ── 5b  Top reasons among early leavers ──────────────────────────────────────
if COL['reason'] in early_vol.columns:
    early_reasons = early_vol[COL['reason']].value_counts().head(10)
    print('\nTop reasons among early voluntary leavers:')
    print(early_reasons.to_string())

    fig, ax = plt.subplots(figsize=(10, 5))
    early_reasons.sort_values().plot(kind='barh', ax=ax, color=BRAND_ORANGE)
    ax.set_title('Top Reasons – Early Voluntary Leavers (< 1 Year)')
    ax.set_xlabel('Count')
    plt.savefig('s5_early_reasons.png', bbox_inches='tight')
    plt.show()

# ── 5c  Area breakdown ───────────────────────────────────────────────────────
if COL['area'] in vol.columns:
    area_early = vol.groupby(COL['area'])['Early_Leaver'].mean().mul(100).round(1)
    print('\nEarly leaver % by Area (voluntary exits only):')
    print(area_early.sort_values(ascending=False).to_string())

# ── 5d  Harrods Band breakdown ───────────────────────────────────────────────
if COL['harrods_band'] in vol.columns:
    band_early = vol.groupby(COL['harrods_band'])['Early_Leaver'].mean().mul(100).round(1)
    print('\nEarly leaver % by Harrods Band:')
    print(band_early.sort_values(ascending=False).to_string())


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

    # Boxplot
    sns.boxplot(data=turnover, x=COL['category'], y=COL['commute_km'],
                ax=axes[0], palette=PALETTE_2, fliersize=3)
    axes[0].set_title('Commute Distance by Off-Board Category')
    axes[0].set_xlabel('')

    # Bar chart – % voluntary by commute band
    if COL['commute_band'] in turnover.columns:
        vol_pct_commute = (
            turnover.groupby(COL['commute_band'])['Is_Voluntary']
                    .mean()
                    .mul(100)
                    .sort_values(ascending=False)
        )
        axes[1].bar(range(len(vol_pct_commute)), vol_pct_commute.values, color=BRAND_BLUE)
        axes[1].set_xticks(range(len(vol_pct_commute)))
        axes[1].set_xticklabels(vol_pct_commute.index, rotation=30, ha='right')
        axes[1].set_title('Voluntary Attrition % by Commute Band')
        axes[1].set_ylabel('% Voluntary')
        axes[1].yaxis.set_major_formatter(mticker.PercentFormatter())

    fig.suptitle('Section 6 – Commute Analysis', fontweight='bold', fontsize=13)
    plt.savefig('s6_commute_analysis.png', bbox_inches='tight')
    plt.show()

# ── 6b  Commute-related resignation reasons ───────────────────────────────────
if COL['reason'] in vol.columns:
    commute_keywords = ['commute', 'travel', 'distance', 'location']
    commute_leavers = vol[
        vol[COL['reason']].str.lower().str.contains('|'.join(commute_keywords), na=False)
    ]
    print(f'\n  Voluntary leavers citing commute-related reasons: {len(commute_leavers):,}')
    if len(commute_leavers) > 0:
        print(commute_leavers[COL['commute_km']].describe().round(2))


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

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, col in zip(axes, pay_cols[:2]):
        sns.boxplot(data=turnover, x=COL['category'], y=col,
                    ax=ax, palette=PALETTE_2, fliersize=3)
        ax.set_title(f'{col.replace("_", " ")} by Category')
        ax.set_xlabel('')

    fig.suptitle('Section 7 – Pay Analysis', fontweight='bold', fontsize=13)
    plt.savefig('s7_pay_boxplots.png', bbox_inches='tight')
    plt.show()

    # Pay by resignation reason
    if COL['reason'] in vol.columns and COL['daily_salary'] in vol.columns:
        pay_by_reason = (
            vol.groupby(COL['reason'])[COL['daily_salary']]
               .median()
               .sort_values()
        )
        fig, ax = plt.subplots(figsize=(10, 7))
        pay_by_reason.plot(kind='barh', ax=ax, color=BRAND_BLUE)
        ax.set_title('Median Daily Salary by Voluntary Off-Board Reason')
        ax.set_xlabel('Median Daily Salary (£)')
        plt.savefig('s7_pay_by_reason.png', bbox_inches='tight')
        plt.show()

    # Pay-dissatisfaction comparison
    if COL['reason'] in vol.columns:
        pay_keywords = ['pay', 'salary', 'compensation', 'wage', 'remuneration']
        pay_leavers    = vol[vol[COL['reason']].str.lower()
                              .str.contains('|'.join(pay_keywords), na=False)]
        non_pay_leavers = vol[~vol[COL['reason']].str.lower()
                               .str.contains('|'.join(pay_keywords), na=False)]

        print(f'\n  Pay-related voluntary leavers : {len(pay_leavers):,}')
        if len(pay_leavers) > 0 and COL['daily_salary'] in vol.columns:
            print(f'  Median salary (pay leavers)   : £{pay_leavers[COL["daily_salary"]].median():,.0f}')
            print(f'  Median salary (other vol)     : £{non_pay_leavers[COL["daily_salary"]].median():,.0f}')

    # Pay by Harrods Band
    if COL['harrods_band'] in turnover.columns and COL['daily_salary'] in turnover.columns:
        pay_band = (
            turnover.groupby([COL['harrods_band'], COL['category']])[COL['daily_salary']]
                    .median()
                    .unstack()
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
# 8. DEPARTMENT & ORGANISATIONAL HOTSPOT ANALYSIS
# =============================================================================
print('\n' + '='*65)
print('SECTION 8 – ORGANISATIONAL HOTSPOT ANALYSIS')
print('='*65)

org_vars = [c for c in [COL['area'], COL['division'], COL['business_unit'],
                         COL['harrods_band'], COL['job_profile']]
            if c in turnover.columns]

# ── 8a  Ranked tables ────────────────────────────────────────────────────────
for var in org_vars:
    vol_by = vol[var].value_counts()
    total_by = turnover[var].value_counts()
    pct_by = (vol_by / total_by * 100).round(1)
    ranked = pd.DataFrame({
        'Voluntary_Leavers': vol_by,
        'Total_Leavers': total_by,
        'Vol_%': pct_by,
    }).sort_values('Voluntary_Leavers', ascending=False).head(15)
    print(f'\nTop 15 voluntary leaver hotspots by {var}:')
    print(ranked.to_string())

# ── 8b  Bar charts ───────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, min(len(org_vars), 3), figsize=(6 * min(len(org_vars), 3), 5))
if len(org_vars) == 1:
    axes = [axes]

for ax, var in zip(axes, org_vars[:3]):
    top_vol = vol[var].value_counts().head(10)
    top_vol.sort_values().plot(kind='barh', ax=ax, color=BRAND_BLUE)
    ax.set_title(f'Voluntary Leavers by {var.replace("_", " ")}')
    ax.set_xlabel('Count')

fig.suptitle('Section 8 – Organisational Hotspots', fontweight='bold', fontsize=13)
plt.savefig('s8_hotspot_bars.png', bbox_inches='tight')
plt.show()

# ── 8c  Heatmap – Area × Division ────────────────────────────────────────────
if COL['area'] in vol.columns and COL['division'] in vol.columns:
    heat_data = (
        vol.groupby([COL['area'], COL['division']])
           .size()
           .unstack(fill_value=0)
    )
    fig, ax = plt.subplots(figsize=(max(10, heat_data.shape[1] * 1.2),
                                    max(6, heat_data.shape[0] * 0.6)))
    sns.heatmap(heat_data, annot=True, fmt='d', cmap='YlOrRd',
                linewidths=0.5, ax=ax, cbar_kws={'label': 'Voluntary Leavers'})
    ax.set_title('Section 8 – Voluntary Leavers Heatmap: Area × Division')
    ax.set_xlabel('Division')
    ax.set_ylabel('Area')
    plt.savefig('s8_heatmap_area_division.png', bbox_inches='tight')
    plt.show()


# =============================================================================
# 9. VOLUNTARY REASON ANALYSIS
# =============================================================================
print('\n' + '='*65)
print('SECTION 9 – VOLUNTARY REASON ANALYSIS')
print('='*65)

if COL['reason'] in vol.columns:
    top_reasons_overall = vol[COL['reason']].value_counts()
    print('\nTop voluntary off-board reasons:')
    print(top_reasons_overall.head(15).to_string())

    breakdown_vars = [c for c in [COL['tenure_band'], COL['age_band'],
                                   COL['harrods_band'], COL['area']]
                      if c in vol.columns]

    for var in breakdown_vars:
        ct = pd.crosstab(vol[var], vol[COL['reason']])

        # ── Heatmap ──────────────────────────────────────────────────────────
        fig, ax = plt.subplots(figsize=(max(12, ct.shape[1] * 1.0),
                                        max(5, ct.shape[0] * 0.5)))
        sns.heatmap(ct, annot=True, fmt='d', cmap='Blues',
                    linewidths=0.3, ax=ax,
                    cbar_kws={'label': 'Voluntary Leavers'})
        ax.set_title(f'Off-Board Reason by {var.replace("_", " ")}')
        ax.set_xlabel('Off-Board Reason')
        ax.set_ylabel(var.replace('_', ' '))
        plt.savefig(f's9_heatmap_{var}.png', bbox_inches='tight')
        plt.show()

        # ── Stacked bar ───────────────────────────────────────────────────────
        ct_pct = ct.div(ct.sum(axis=1), axis=0) * 100
        top_reasons_list = top_reasons_overall.head(8).index.tolist()
        ct_pct_top = ct_pct[[c for c in top_reasons_list if c in ct_pct.columns]]

        fig, ax = plt.subplots(figsize=(10, 5))
        ct_pct_top.plot(kind='bar', stacked=True, ax=ax, colormap='tab10', edgecolor='white')
        ax.set_title(f'Voluntary Reason Breakdown by {var.replace("_", " ")} (%)')
        ax.set_xlabel(var.replace('_', ' '))
        ax.set_ylabel('% of Voluntary Leavers')
        ax.legend(title='Reason', bbox_to_anchor=(1.05, 1), loc='upper left', frameon=False)
        ax.tick_params(axis='x', rotation=30)
        plt.savefig(f's9_stackedbar_{var}.png', bbox_inches='tight')
        plt.show()


# =============================================================================
# 10. ATTRITION PERSONAS
# =============================================================================
print('\n' + '='*65)
print('SECTION 10 – ATTRITION PERSONAS')
print('='*65)

PERSONA_MAP = {
    'Career':             ['Development Opportunities', 'Hired by Competitor', 'Return to Education'],
    'Employee Experience':['Culture', 'Working Environment'],
    'Lifestyle':          ['Commute Time', 'Relocation', 'Work Life Balance'],
    'Personal':           ['Family Reasons', 'Ill Health', 'Visa Expiry'],
}

if COL['reason'] in vol.columns:
    # Build persona column using partial string matching
    def assign_persona(reason):
        if pd.isna(reason):
            return 'Other'
        reason_lower = str(reason).lower()
        for persona, keywords in PERSONA_MAP.items():
            for kw in keywords:
                if kw.lower() in reason_lower:
                    return persona
        return 'Other'

    vol = vol.copy()
    vol['Persona'] = vol[COL['reason']].apply(assign_persona)

    persona_counts = vol['Persona'].value_counts()
    print('\nPersona distribution:')
    print(persona_counts.to_string())

    # Profile numeric stats per persona
    profile_cols = [c for c in [COL['age'], COL['tenure'],
                                  COL['daily_salary'], COL['commute_km']]
                    if c in vol.columns]

    persona_profile = vol.groupby('Persona')[profile_cols].agg(['mean', 'median']).round(2)
    print('\nPersona profiles (mean | median):')
    print(persona_profile.to_string())

    # ── Radar / violin plots ──────────────────────────────────────────────────
    if profile_cols:
        fig, axes = plt.subplots(1, len(profile_cols),
                                  figsize=(4 * len(profile_cols), 5))
        if len(profile_cols) == 1:
            axes = [axes]

        palette_persona = sns.color_palette('Set2', n_colors=len(vol['Persona'].unique()))
        for ax, col in zip(axes, profile_cols):
            sns.violinplot(data=vol, x='Persona', y=col,
                           ax=ax, palette=palette_persona, inner='box', linewidth=0.8)
            ax.set_title(col.replace('_', ' '))
            ax.set_xlabel('')
            ax.tick_params(axis='x', rotation=30)

        fig.suptitle('Section 10 – Attrition Persona Profiles', fontweight='bold', fontsize=13)
        plt.savefig('s10_persona_profiles.png', bbox_inches='tight')
        plt.show()

    # Donut chart – persona share
    fig, ax = plt.subplots(figsize=(7, 7))
    wedges, texts, autotexts = ax.pie(
        persona_counts.values,
        labels=persona_counts.index,
        autopct='%1.1f%%',
        startangle=90,
        wedgeprops={'width': 0.55, 'edgecolor': 'white', 'linewidth': 2},
        colors=sns.color_palette('Set2', n_colors=len(persona_counts)),
    )
    for at in autotexts:
        at.set_fontweight('bold')
    ax.set_title('Section 10 – Voluntary Attrition Personas', fontweight='bold')
    plt.savefig('s10_persona_donut.png', bbox_inches='tight')
    plt.show()

    # Harrods Band by Persona heatmap
    if COL['harrods_band'] in vol.columns:
        persona_band = pd.crosstab(vol['Persona'], vol[COL['harrods_band']])
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.heatmap(persona_band, annot=True, fmt='d', cmap='Purples',
                    linewidths=0.3, ax=ax)
        ax.set_title('Personas by Harrods Band')
        plt.savefig('s10_persona_band_heatmap.png', bbox_inches='tight')
        plt.show()


# =============================================================================
# 11. STATISTICAL DRIVER ANALYSIS (Chi-Square + Cramér's V)
# =============================================================================
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
print('\nChi-Square Driver Analysis Results:')
print(chi_df.to_string(index=False))

fig, ax = plt.subplots(figsize=(8, 4))
bars = ax.barh(chi_df['Variable'], chi_df['Cramers_V'],
               color=[BRAND_BLUE if s == 'Yes' else BRAND_ORANGE
                      for s in chi_df['Significant']])
ax.axvline(0.1, color='gray', linestyle='--', linewidth=0.8, label='Small effect (0.1)')
ax.axvline(0.3, color='gray', linestyle=':',  linewidth=0.8, label='Medium effect (0.3)')
ax.set_title("Section 11 – Cramér's V Effect Size by Variable")
ax.set_xlabel("Cramér's V")
ax.legend(frameon=False, fontsize=8)
plt.savefig('s11_chi_square_effect_sizes.png', bbox_inches='tight')
plt.show()


# =============================================================================
# 12. LOGISTIC REGRESSION
# =============================================================================
print('\n' + '='*65)
print('SECTION 12 – LOGISTIC REGRESSION')
print('='*65)

lr_num_cols = [c for c in [COL['age'], COL['tenure'],
                             COL['daily_salary'], COL['commute_km']]
               if c in turnover.columns]

lr_cat_cols = [c for c in [COL['harrods_band']] if c in turnover.columns]

lr_df = turnover[lr_num_cols + lr_cat_cols + ['Is_Voluntary']].dropna()

# One-hot encode categorical predictors
if lr_cat_cols:
    lr_df = pd.get_dummies(lr_df, columns=lr_cat_cols, drop_first=True)

feature_cols = [c for c in lr_df.columns if c != 'Is_Voluntary']

X = lr_df[feature_cols].astype(float)
y = lr_df['Is_Voluntary']

# ── 12a  Multicollinearity check (VIF) ───────────────────────────────────────
X_vif = sm.add_constant(X)
vif_data = pd.DataFrame({
    'Feature': X_vif.columns,
    'VIF': [variance_inflation_factor(X_vif.values, i) for i in range(X_vif.shape[1])]
}).sort_values('VIF', ascending=False)
print('\nVariance Inflation Factors:')
print(vif_data.to_string(index=False))

# ── 12b  Fit model ───────────────────────────────────────────────────────────
X_const = sm.add_constant(X)
model = sm.Logit(y, X_const).fit(disp=False)
print('\nLogistic Regression Summary:')
print(model.summary2())

# ── 12c  Odds Ratios ──────────────────────────────────────────────────────────
odds_df = pd.DataFrame({
    'Feature'   : model.params.index,
    'Coef'      : model.params.values.round(4),
    'Odds_Ratio': np.exp(model.params.values).round(4),
    'p_value'   : model.pvalues.values.round(4),
    'CI_Lower'  : np.exp(model.conf_int()[0].values).round(4),
    'CI_Upper'  : np.exp(model.conf_int()[1].values).round(4),
}).sort_values('Odds_Ratio', ascending=False)

print('\nOdds Ratio Interpretation Table (for HR stakeholders):')
print(odds_df.to_string(index=False))

# Forest plot of odds ratios
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
# 13. CORRELATION ANALYSIS
# =============================================================================
print('\n' + '='*65)
print('SECTION 13 – CORRELATION ANALYSIS')
print('='*65)

corr_cols = [c for c in [COL['age'], COL['tenure'], COL['daily_salary'],
                           COL['base_pay'], COL['commute_km']]
             if c in turnover.columns]

corr_matrix = turnover[corr_cols].corr()
print('\nCorrelation Matrix:')
print(corr_matrix.round(3).to_string())

fig, ax = plt.subplots(figsize=(8, 6))
mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm',
            vmin=-1, vmax=1, center=0,
            linewidths=0.5, ax=ax, square=True,
            cbar_kws={'label': 'Pearson r'},
            xticklabels=[c.replace('_', ' ') for c in corr_cols],
            yticklabels=[c.replace('_', ' ') for c in corr_cols])
ax.set_title('Section 13 – Correlation Matrix (Numeric Variables)')
plt.savefig('s13_correlation_heatmap.png', bbox_inches='tight')
plt.show()

print("""
Interpretation note:
  Correlations describe the strength of LINEAR association between variables.
  They do NOT imply that one variable causes another.
  High correlation between salary and tenure, for example, likely reflects
  pay progression over time rather than salary directly causing retention.
  Always contextualise statistical findings with domain knowledge.
""")


# =============================================================================
# 14. EXECUTIVE SUMMARY OUTPUT
# =============================================================================
print('\n' + '='*65)
print('SECTION 14 – EXECUTIVE SUMMARY')
print('='*65)

# ── Auto-derive key facts ─────────────────────────────────────────────────────
top3_reasons = (
    vol[COL['reason']].value_counts().head(3).index.tolist()
    if COL['reason'] in vol.columns else ['N/A']
)

top_area = (
    vol[COL['area']].value_counts().idxmax()
    if COL['area'] in vol.columns else 'N/A'
)

top_division = (
    vol[COL['division']].value_counts().idxmax()
    if COL['division'] in vol.columns else 'N/A'
)

early_pct_str = f'{pct_early:.1f}%' if 'pct_early' in dir() else 'N/A'

significant_drivers = (
    chi_df[chi_df['Significant'] == 'Yes']['Variable'].tolist()
    if 'chi_df' in dir() else []
)

top_sig_driver = significant_drivers[0] if significant_drivers else 'N/A'

try:
    top_lr_predictor = odds_plot.iloc[0]['Feature'].replace('_', ' ')
except Exception:
    top_lr_predictor = 'N/A'

print(f"""
╔══════════════════════════════════════════════════════════════╗
║            EMPLOYEE ATTRITION – EXECUTIVE SUMMARY           ║
╚══════════════════════════════════════════════════════════════╝

  HEADLINE METRICS
  ─────────────────────────────────────────────────────────────
  • Total leavers         : {total:,}
  • Voluntary attrition   : {total_vol:,}  ({pct_vol:.1f}%)
  • Involuntary attrition : {total_invol:,}  ({pct_invol:.1f}%)
  • Early leavers (<1yr)  : {len(early_vol):,}  ({early_pct_str} of voluntary exits)

  TOP 5 VOLUNTARY ATTRITION INSIGHTS
  ─────────────────────────────────────────────────────────────
  1. The top voluntary off-board reasons are:
     {', '.join(top3_reasons)}
  2. Early leavers (< 1 year tenure) account for {early_pct_str}
     of voluntary exits – a key retention risk.
  3. The highest voluntary leaver concentration by Area is:
     {top_area}
  4. The highest voluntary leaver concentration by Division is:
     {top_division}
  5. The strongest statistical predictor of voluntary attrition
     in the logistic regression is: {top_lr_predictor}

  BIGGEST TENURE RISKS
  ─────────────────────────────────────────────────────────────
  • Employees leaving within their first year represent a
    significant proportion of voluntary exits.
  • Focus onboarding and early engagement interventions on
    the 0–12 month window.

  BIGGEST ORGANISATIONAL HOTSPOTS
  ─────────────────────────────────────────────────────────────
  • Area:     {top_area}
  • Division: {top_division}
  • Review management practices, workload, and culture signals
    in these areas as a priority.

  MOST COMMON RESIGNATION REASONS
  ─────────────────────────────────────────────────────────────
  {chr(10).join(f'  • {r}' for r in top3_reasons)}

  SIGNIFICANT STATISTICAL FINDINGS
  ─────────────────────────────────────────────────────────────
  • The following variables are significantly associated with
    voluntary attrition (Chi-Square, p < 0.05):
    {', '.join(significant_drivers) if significant_drivers else 'See Section 11'}
  • Cramér's V identifies {top_sig_driver} as the
    strongest categorical driver.

  POTENTIAL RETENTION ACTIONS
  ─────────────────────────────────────────────────────────────
  1. Strengthen structured onboarding and 90-day check-ins
     to reduce early leaver rates.
  2. Review pay competitiveness, particularly in grades and
     roles where pay-related exits are elevated.
  3. Evaluate remote/hybrid working options for employees
     with long commutes flagging commute as a factor.
  4. Invest in career development pathways, especially for
     high-risk tenure bands (0–3 years).
  5. Conduct targeted stay interviews in hotspot Areas and
     Divisions to surface localised engagement issues.
  6. Monitor culture and working environment signals through
     regular pulse surveys in high-exit Business Units.

══════════════════════════════════════════════════════════════
  All charts saved as PNG files in the working directory.
  Analysis complete.
══════════════════════════════════════════════════════════════
""")
