#!/usr/bin/env python3
"""
Build a unified NISR post-harvest modeling dataset for Rwanda (2024-2025).

Inputs expected in INPUT_DIR (default: script directory):
  - Dataset.zip       : NISR SAS 2024 raw modules (Production + Agricultural Practice + others)
  - all_seasons.zip   : user's already-merged NISR SAS 2025 Production + Agricultural Practice data
  - Climate.zip       : CHIRPS 2024-2025, NASA POWER 2024-2025 RH2M/T2M, Rwanda ADM2 boundaries
  - Optional standalone climate files (e.g. 2023 CHIRPS/NASA POWER) placed in INPUT_DIR

Outputs:
  - postharvest_regression_model_ready_2024_2025.csv
  - postharvest_regression_model_dictionary.csv
  - postharvest_regression_model_manifest.json
  - unified_postharvest_data_dictionary.csv
  - unified_postharvest_quality_report.txt
  - district_daily_climate_features.csv

Scientific choices:
  * Final observation grain is farmer x crop x season x year because post-harvest loss outcomes
    are crop-level and repeated across multiple plots for the same farmer/crop.
  * Main regression target = storage + processing + packaging loss rate (% of crop production).
  * Transport and sales losses are retained as descriptive-only fields, not main ML targets.
  * Negative losses and physically impossible actionable losses (> total reported crop production)
    are not guessed/corrected; they are flagged and removed from model-ready target values.
  * Climate is matched to an estimated harvest-date midpoint derived from the survey harvest-period
    category. "Other season" uses a season midpoint fallback and is explicitly flagged.
  * Weather after harvest is historical here; in a live predictive application it must be replaced
    by weather forecasts for the same window.

Run:
    python build_unified_postharvest_dataset.py
or:
    python build_postharvest_model_ready_corrected.py --input-dir /path/to/files --output-dir /path/to/output
"""

from __future__ import annotations

import argparse
import json
import math
import re
import shutil
import tempfile
import warnings
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.mask import mask as rio_mask
from scipy.spatial import cKDTree


# ----------------------------- Configuration -----------------------------

PROD24_MAP = {
    'Segment_ID':'segment_id', 's1q1':'province', 's1q2':'district', 's1q3':'stratum',
    's1q6':'farmer_id', 's1q7':'farmer_type', 's1q8':'farmer_gender', 's1q9':'farmer_age',
    's2q1':'plot_number', 's2q2':'plot_area_sqm', 's2q3':'num_crops_in_plot',
    's2q4':'crop_name', 's2q4_o':'crop_name_other', 'CropCategory':'crop_category',
    's2q8':'expected_harvest_period', 's2q9':'used_improved_seed', 's2q11':'seed_type_sown',
    's2q19_3':'qty_harvested_kg', 's2q20_2':'qty_remaining_to_harvest_kg',
    's2q21':'total_harvest_kg', 's2q22':'total_production_qty',
    's2q23':'production_vs_prev_season', 's2q25':'qty_transformed_kg',
    's2q26':'qty_sold_kg', 's2q29':'qty_hh_consumption_kg',
    's2q34':'qty_reserved_seed_kg', 's2q35':'qty_used_fodder_kg',
    's2q36':'qty_stored_future_use_kg', 's2q37':'storage_type',
    's2q38':'qty_stored_public_kg', 's2q41':'loss_stolen_kg',
    's2q42':'loss_pest_damage_kg', 's2q43':'loss_birds_animals_kg',
    's2q44':'loss_stalks_fallen_kg', 's2q45':'loss_harvesting_kg',
    's2q46':'loss_transport_kg', 's2q47':'loss_storage_kg',
    's2q48':'loss_processing_kg', 's2q49':'loss_packaging_kg', 's2q50':'loss_sales_kg',
    's3q1':'used_organic_fert_any_plot', 's3q3':'used_organic_fert_this_plot',
    's3q7':'used_inorganic_fert_any_plot', 's3q9':'used_inorganic_fert_this_plot',
    's3q16':'used_micronutrients_any_plot', 's3q17':'used_micronutrients_this_plot',
    's3q18':'used_pesticide_any_plot', 's3q19':'used_pesticide_this_plot',
    's4q1':'erosion_degree', 's4q2':'anti_erosion_any_plot',
    's4q3':'anti_erosion_this_plot', 's4q4':'anti_erosion_done_this_season',
    's4q6':'land_consolidated_site', 's4q8':'used_mech_equipment_any_plot',
    's4q9':'used_mech_equipment_this_plot', 's4q14':'practiced_irrigation_any_plot',
    's4q15':'plot_irrigated_this_season', 'plot_weight':'plot_weight',
}

PRAC24_MAP = {
    'Segment_ID':'segment_id', 's1q6':'farmer_id', 's2q1':'plot_number',
    's4q13':'hired_labor_cost_rwf', 's4q16':'irrigation_technique',
    's4q17_1':'irrigation_water_source', 's4q18_1':'irrigation_tool',
    's4q19':'irrigation_cost_rwf',
}

PROD25_MAP = {
    'Segment_ID':'segment_id', 'province':'province', 'district':'district', 'stratum':'stratum',
    'farmer_id':'farmer_id', 'farmer_type':'farmer_type', 'farmer_gender':'farmer_gender',
    'farmer_age':'farmer_age', 'plot_number':'plot_number', 'plot_area_sqm':'plot_area_sqm',
    'num_crops_in_plot':'num_crops_in_plot', 'crop_name_1':'crop_name',
    'crop_name_other':'crop_name_other', 'crop_category':'crop_category',
    'expected_harvest_period':'expected_harvest_period', 'used_improved_seed':'used_improved_seed',
    'seed_type_sown':'seed_type_sown', 'qty_harvested_kg':'qty_harvested_kg',
    'qty_remaining_to_harvest_kg':'qty_remaining_to_harvest_kg',
    'total_harvest_kg':'total_harvest_kg', 'total_production_qty':'total_production_qty',
    'production_vs_prev_season':'production_vs_prev_season', 'qty_transformed':'qty_transformed_kg',
    'qty_sold':'qty_sold_kg', 'qty_hh_consumption':'qty_hh_consumption_kg',
    'qty_reserved_seed':'qty_reserved_seed_kg', 'qty_used_fodder':'qty_used_fodder_kg',
    'qty_stored_future_use':'qty_stored_future_use_kg', 'storage_type':'storage_type',
    'qty_stored_public_kg':'qty_stored_public_kg', 'loss_stolen_kg':'loss_stolen_kg',
    'loss_pest_damage_kg':'loss_pest_damage_kg', 'loss_birds_animals_kg':'loss_birds_animals_kg',
    'loss_stalks_fallen_kg':'loss_stalks_fallen_kg', 'loss_harvesting_kg':'loss_harvesting_kg',
    'loss_transport_kg':'loss_transport_kg', 'loss_storage_kg':'loss_storage_kg',
    'loss_processing_kg':'loss_processing_kg', 'loss_packaging_kg':'loss_packaging_kg',
    'loss_sales_kg':'loss_sales_kg', 'used_organic_fert_any_plot':'used_organic_fert_any_plot',
    'used_organic_fert_this_plot':'used_organic_fert_this_plot',
    'used_inorganic_fert_any_plot':'used_inorganic_fert_any_plot',
    'used_inorganic_fert_this_plot':'used_inorganic_fert_this_plot',
    'used_micronutrients_any_plot':'used_micronutrients_any_plot',
    'used_micronutrients_this_plot':'used_micronutrients_this_plot',
    'used_pesticide_any_plot':'used_pesticide_any_plot',
    'used_pesticide_this_plot':'used_pesticide_this_plot', 'erosion_degree':'erosion_degree',
    'anti_erosion_any_plot':'anti_erosion_any_plot', 'anti_erosion_this_plot':'anti_erosion_this_plot',
    'anti_erosion_done_this_season':'anti_erosion_done_this_season',
    'land_consolidated_site':'land_consolidated_site',
    'used_mech_equipment_any_plot':'used_mech_equipment_any_plot',
    'used_mech_equipment_this_plot':'used_mech_equipment_this_plot',
    'practiced_irrigation_any_plot':'practiced_irrigation_any_plot',
    'plot_irrigated_this_season':'plot_irrigated_this_season', 'plot_weight':'plot_weight',
    'irrigation_technique':'irrigation_technique',
    'irrigation_water_source':'irrigation_water_source', 'irrigation_tool':'irrigation_tool',
    'irrigation_cost_rwf':'irrigation_cost_rwf',
}

DISTRICT_MAP = {'Nyarugenege':'Nyarugenge'}
CROP_MAP = {
    'Mandaline':'Mandoline', 'Sun flower':'Sunflower', 'sugar Beet':'Sugar beet',
    'Jack Fruits':'Jack fruit', 'Other seasonal crops(specify).':'Other seasonal crops (specify)',
    'Other seasonal crops(specify)':'Other seasonal crops (specify)',
    'Other annual crops (specify).':'Other annual crops (specify)',
    'Other perennial crop (Specify).':'Other perennial crop (specify)',
    'Other fodder crop (specify).':'Other fodder crop (specify)',
    'Other fodder crop (specify)':'Other fodder crop (specify)',
    'Other seasonal vegetables(specify)':'Other seasonal vegetables (specify)',
}

HARVEST_MIDPOINTS = {
    'Before 01/12': (11,15), 'Between 01-15/12': (12,8), 'Between 16-31/12': (12,23),
    'Between 01-15/01': (1,8), 'Between 16-31/01': (1,23), 'Between 01-28/02': (2,14),
    'After Feb': (3,15), 'Before 01/05': (4,15), 'Between 01-15/05': (5,8),
    'Between 15-31/05': (5,23), 'Between 01-15/06': (6,8), 'Between 16-30/06': (6,23),
    'Between 01-15/07': (7,8), 'Between 16-31/07': (7,23), 'Between 01-31/08': (8,15),
    'After August': (9,15), 'Before 01/08': (7,15), 'Between 01-15/08': (8,8),
    'Between 16-31/08': (8,23), 'Between 01-15/09': (9,8), 'Between 16-30/09': (9,23),
    'After 30/09': (10,15),
}
SEASON_FALLBACK = {'A':(1,15), 'B':(6,30), 'C':(9,15)}

PLOT_BOOL_COLS = [
    'used_improved_seed', 'used_organic_fert_this_plot', 'used_inorganic_fert_this_plot',
    'used_micronutrients_this_plot', 'used_pesticide_this_plot', 'anti_erosion_this_plot',
    'used_mech_equipment_this_plot', 'plot_irrigated_this_season',
]
FARMER_BOOL_COLS = [
    'used_organic_fert_any_plot', 'used_inorganic_fert_any_plot', 'used_micronutrients_any_plot',
    'used_pesticide_any_plot', 'anti_erosion_any_plot', 'anti_erosion_done_this_season',
    'land_consolidated_site', 'used_mech_equipment_any_plot', 'practiced_irrigation_any_plot',
]


# ----------------------------- Utilities -----------------------------

def extract_zip(path: Path, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path) as z:
        z.extractall(out)


def clean_text_series(s: pd.Series) -> pd.Series:
    out = s.astype('string')
    out = out.str.replace('\u00a0', ' ', regex=False)
    out = out.str.replace(r'\s+', ' ', regex=True).str.strip()
    out = out.mask(out.isin(['', 'nan', 'NaN', 'None', '<NA>']))
    return out


def normalize_id_series(s: pd.Series) -> pd.Series:
    out = s.astype('string').str.strip().str.replace(r'\.0$', '', regex=True)
    return out


def read_dta_with_value_labels(path: Path) -> pd.DataFrame:
    """Read Stata while applying value labels without pandas categorical uniqueness problems."""
    reader = pd.io.stata.StataReader(path)
    df = reader.read(convert_categoricals=False)
    labels = reader.value_labels()
    variable_to_label = dict(zip(reader._varlist, reader._lbllist))
    for col, label_name in variable_to_label.items():
        if col in df.columns and label_name and label_name in labels:
            mapped = df[col].map(labels[label_name])
            if mapped.notna().any():
                df[col] = mapped.where(mapped.notna(), df[col])
    return df


def clean_common(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    text_cols = [c for c in df.columns if df[c].dtype == 'object' or str(df[c].dtype).startswith('string')]
    for c in text_cols:
        df[c] = clean_text_series(df[c])

    if 'district' in df:
        df['district'] = df['district'].replace(DISTRICT_MAP)
    if 'crop_name' in df:
        df['crop_name'] = df['crop_name'].replace(CROP_MAP)
    if 'crop_category' in df:
        df['crop_category'] = df['crop_category'].replace(CROP_MAP)
    if 'expected_harvest_period' in df:
        df['expected_harvest_period'] = (df['expected_harvest_period']
            .str.replace('Other season (applicable  for all crops)', 'Other season (applicable for all crops)', regex=False)
            .str.replace('Between 16- 31/01', 'Between 16-31/01', regex=False)
            .str.replace('Between 01- 15/06', 'Between 01-15/06', regex=False)
            .str.replace('Between 16 -30/06', 'Between 16-30/06', regex=False)
            .str.replace('Between  01-31/08', 'Between 01-31/08', regex=False)
            .str.replace('Between 16- 31/08', 'Between 16-31/08', regex=False)
            .str.replace('Between 16 -30/09', 'Between 16-30/09', regex=False))

    for c in ['storage_type', 'production_vs_prev_season']:
        if c in df:
            df[c] = df[c].mask(df[c].astype('string').str.match(r'^\d+(\.0)?$', na=False))

    numeric_cols = [
        'farmer_age','plot_number','plot_area_sqm','num_crops_in_plot','qty_harvested_kg',
        'qty_remaining_to_harvest_kg','total_harvest_kg','total_production_qty','qty_transformed_kg',
        'qty_sold_kg','qty_hh_consumption_kg','qty_reserved_seed_kg','qty_used_fodder_kg',
        'qty_stored_future_use_kg','qty_stored_public_kg','loss_stolen_kg','loss_pest_damage_kg',
        'loss_birds_animals_kg','loss_stalks_fallen_kg','loss_harvesting_kg','loss_transport_kg',
        'loss_storage_kg','loss_processing_kg','loss_packaging_kg','loss_sales_kg','plot_weight',
        'hired_labor_cost_rwf','irrigation_cost_rwf',
    ]
    for c in numeric_cols:
        if c in df:
            df[c] = pd.to_numeric(df[c], errors='coerce')

    # Conservative age handling: >100 treated invalid; under 15 only flagged, not erased.
    if 'farmer_age' in df:
        df['farmer_age_over100_flag'] = (df['farmer_age'] > 100).astype('int8')
        df['farmer_age_under15_flag'] = (df['farmer_age'] < 15).astype('int8')
        df.loc[df['farmer_age'] > 100, 'farmer_age'] = np.nan

    # Negative physical losses are invalid. Keep a flag and set the value missing.
    loss_cols = [c for c in df.columns if c.startswith('loss_') and c.endswith('_kg')]
    for c in loss_cols:
        neg = df[c] < 0
        df[c + '_was_negative'] = neg.astype('int8')
        df.loc[neg, c] = np.nan

    return df


# ----------------------------- NISR preparation -----------------------------

def prepare_2024_season(dataset_root: Path, season: str, qa: dict) -> pd.DataFrame:
    sdir = dataset_root / f'Season {season}'
    prod_path = sdir / f'Rwa_raw_Season{season}2024_Production.dta'
    prac_path = sdir / f'Rwa_raw_Season{season}2024_Agricultural_practice.dta'
    prod = read_dta_with_value_labels(prod_path)
    prac = read_dta_with_value_labels(prac_path)

    prod_sel = prod[[c for c in PROD24_MAP if c in prod.columns]].rename(columns=PROD24_MAP)
    prac_sel = prac[[c for c in PRAC24_MAP if c in prac.columns]].rename(columns=PRAC24_MAP)
    keys = ['segment_id','farmer_id','plot_number']

    dup_practice = int(prac_sel.duplicated(keys).sum())
    if dup_practice:
        raise ValueError(f'2024 Season {season}: Agricultural Practice has {dup_practice} duplicate merge keys.')

    merged = prod_sel.merge(prac_sel, on=keys, how='left', validate='many_to_one', indicator=True)
    coverage = float((merged['_merge'] == 'both').mean())
    qa[f'2024_{season}_production_rows'] = len(prod)
    qa[f'2024_{season}_practice_rows'] = len(prac)
    qa[f'2024_{season}_practice_merge_coverage'] = coverage
    merged = merged.drop(columns='_merge')
    merged['year'] = 2024
    merged['season'] = season
    return clean_common(merged)


def prepare_2025_from_user_merged(path: Path, qa: dict) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    qa['2025_user_merged_rows'] = len(df)
    qa['2025_user_merged_columns'] = len(df.columns)

    # Two alternative columns in the user's merged file represent the same questionnaire variable.
    if 'used_inorganic_fert_this_plot' in df.columns and 's3q9used_inorganic_fert_this_plot' in df.columns:
        df['used_inorganic_fert_this_plot'] = df['used_inorganic_fert_this_plot'].where(
            df['used_inorganic_fert_this_plot'].notna(), df['s3q9used_inorganic_fert_this_plot'])

    sel = df[[c for c in PROD25_MAP if c in df.columns]].rename(columns=PROD25_MAP)
    sel['year'] = 2025
    sel['season'] = df['season'].values
    return clean_common(sel)


def validate_2025_raw_b(input_dir: Path, qa: dict) -> None:
    prod_path = input_dir / 'Rwa_raw_SeasonB2025_Production.csv'
    prac_path = input_dir / 'Rwa_raw_SeasonB2025_Agricultural_practice.csv'
    if not prod_path.exists() or not prac_path.exists():
        qa['2025_B_raw_validation'] = 'raw B files not present; skipped'
        return
    prod = pd.read_csv(prod_path, sep='\t', low_memory=False)
    prac = pd.read_csv(prac_path, sep='\t', low_memory=False)
    keys = ['Segment_ID','s1q6','s2q1']
    qa['2025_B_raw_production_rows'] = len(prod)
    qa['2025_B_raw_practice_rows'] = len(prac)
    qa['2025_B_raw_practice_duplicate_keys'] = int(prac.duplicated(keys).sum())
    chk = prod[keys].merge(prac[keys].drop_duplicates(), on=keys, how='left', indicator=True)
    qa['2025_B_raw_practice_merge_coverage'] = float((chk['_merge'] == 'both').mean())


def collapse_farmer_crop(df: pd.DataFrame) -> pd.DataFrame:
    """Collapse plot/crop rows to the scientifically correct farmer x crop x season x year grain."""
    d = df.copy()
    d['segment_id'] = normalize_id_series(d['segment_id'])
    d['farmer_id'] = normalize_id_series(d['farmer_id'])
    d['plot_number'] = normalize_id_series(d['plot_number'])
    group = ['year','season','segment_id','farmer_id','crop_name']

    for c in PLOT_BOOL_COLS + FARMER_BOOL_COLS:
        if c in d.columns:
            d[c + '__yn'] = clean_text_series(d[c]).map({'Yes':1.0,'No':0.0})

    g = d.groupby(group, dropna=False, sort=False)
    cat_cols = [c for c in [
        'province','district','stratum','farmer_type','farmer_gender','crop_category',
        'expected_harvest_period','seed_type_sown','production_vs_prev_season','storage_type',
        'erosion_degree','irrigation_technique'
    ] if c in d.columns]
    base = g[cat_cols].first().reset_index()

    aggregations = {
        'farmer_age':'median', 'plot_number':'nunique', 'plot_area_sqm':'sum', 'num_crops_in_plot':'mean',
        'qty_harvested_kg':'sum', 'qty_remaining_to_harvest_kg':'sum', 'total_harvest_kg':'sum',
        'total_production_qty':'max', 'qty_stored_future_use_kg':'max',
        'loss_transport_kg':'max', 'loss_storage_kg':'max', 'loss_processing_kg':'max',
        'loss_packaging_kg':'max', 'loss_sales_kg':'max', 'plot_weight':'mean',
    }
    aggregations = {k:v for k,v in aggregations.items() if k in d.columns}
    num = g.agg(aggregations).reset_index()
    out = base.merge(num, on=group, how='outer', validate='one_to_one')
    out = out.rename(columns={
        'plot_number':'n_plots_crop', 'plot_area_sqm':'crop_plot_area_sqm_total',
        'plot_weight':'survey_weight_mean'
    })
    out['crop_plot_area_ha_total'] = out['crop_plot_area_sqm_total'] / 10000.0

    plot_bool_cols = [c for c in PLOT_BOOL_COLS if c + '__yn' in d.columns]
    if plot_bool_cols:
        tmp = g[[c + '__yn' for c in plot_bool_cols]].mean().reset_index()
        rename = {}
        for c in plot_bool_cols:
            if c.endswith('_this_plot'):
                new = c[:-10] + '_share_yes'
            else:
                new = c + '_share_yes'
            rename[c + '__yn'] = new
        out = out.merge(tmp.rename(columns=rename), on=group, how='left', validate='one_to_one')

    farmer_bool_cols = [c for c in FARMER_BOOL_COLS if c + '__yn' in d.columns]
    if farmer_bool_cols:
        tmp = g[[c + '__yn' for c in farmer_bool_cols]].max().reset_index()
        tmp = tmp.rename(columns={c + '__yn': c + '_yes' for c in farmer_bool_cols})
        out = out.merge(tmp, on=group, how='left', validate='one_to_one')

    # Critical source consistency checks. Losses should be repeated, not summed, across plots.
    critical = [c for c in ['total_production_qty','loss_storage_kg','loss_processing_kg','loss_packaging_kg','loss_transport_kg'] if c in d]
    if critical:
        nu = g[critical].nunique(dropna=True).reset_index().rename(columns={c:f'{c}_source_nunique' for c in critical})
        out = out.merge(nu, on=group, how='left', validate='one_to_one')

    flag_cols = [c for c in d.columns if c.endswith('_was_negative') or c in ['farmer_age_over100_flag','farmer_age_under15_flag']]
    if flag_cols:
        fl = g[flag_cols].max().reset_index()
        out = out.merge(fl, on=group, how='left', validate='one_to_one')

    out['farmer_crop_obs_id'] = (
        out['year'].astype(str) + '_' + out['season'].astype(str) + '_' + out['segment_id'].astype(str) + '_' +
        out['farmer_id'].astype(str) + '_' + out['crop_name'].astype(str).str.replace(r'\W+', '_', regex=True).str.strip('_')
    )
    return out


def clean_model_targets(df: pd.DataFrame) -> pd.DataFrame:
    """Create the continuous regression target conservatively.

    Negative source losses are anomalies: never abs(), never silently zero.
    Farmer-crop groups with any negative actionable-loss source value, inconsistent
    repeated production/loss values, or physically impossible loss > production are
    excluded from regression target construction.
    """
    out = df.copy()
    targets = ['loss_storage_kg','loss_processing_kg','loss_packaging_kg']

    neg_flag_cols = [f'{c}_was_negative' for c in targets if f'{c}_was_negative' in out.columns]
    any_negative = (
        out[neg_flag_cols].fillna(0).max(axis=1).gt(0)
        if neg_flag_cols else pd.Series(False, index=out.index)
    )
    out['actionable_target_invalid_negative_source'] = any_negative.astype('int8')

    consistency_cols = [
        'total_production_qty_source_nunique',
        'loss_storage_kg_source_nunique',
        'loss_processing_kg_source_nunique',
        'loss_packaging_kg_source_nunique',
    ]
    inconsistent = pd.Series(False, index=out.index)
    for c in consistency_cols:
        if c in out.columns:
            inconsistent = inconsistent | out[c].fillna(0).gt(1)
    out['actionable_target_invalid_source_inconsistency'] = inconsistent.astype('int8')

    # If source evidence is ambiguous, do not guess the target.
    source_invalid = any_negative | inconsistent
    out.loc[source_invalid, targets] = np.nan

    complete = out[targets].notna().all(axis=1)
    target_sum = out[targets].sum(axis=1, min_count=3)
    out['actionable_loss_kg'] = target_sum

    impossible = (
        complete
        & out['total_production_qty'].notna()
        & (out['total_production_qty'] >= 0)
        & (target_sum > out['total_production_qty'])
    )
    out['actionable_target_invalid_exceeds_production'] = impossible.astype('int8')
    out.loc[impossible, targets + ['actionable_loss_kg']] = np.nan

    complete2 = out[targets].notna().all(axis=1)
    target_sum2 = out[targets].sum(axis=1, min_count=3)

    # Internal QA only; this binary field is NOT exported to the regression training CSV.
    out['actionable_loss_any'] = np.where(complete2, (target_sum2 > 0).astype(float), np.nan)

    # Main regression target: storage + processing + packaging loss as % of production.
    out['actionable_loss_rate_pct'] = np.where(
        complete2 & (out['total_production_qty'] > 0),
        100.0 * target_sum2 / out['total_production_qty'],
        np.nan,
    )

    # Descriptive only; transport and sales are not part of the regression target.
    out['reported_postharvest_loss_kg_all_stages'] = out[
        ['loss_transport_kg','loss_storage_kg','loss_processing_kg','loss_packaging_kg','loss_sales_kg']
    ].sum(axis=1, min_count=1)
    return out


# ----------------------------- Climate preparation -----------------------------

def read_power_csv(path: Path) -> pd.DataFrame:
    with path.open('r', encoding='utf-8') as f:
        lines = f.readlines()
    end_header = next(i for i, line in enumerate(lines) if line.strip() == '-END HEADER-')
    return pd.read_csv(path, skiprows=end_header + 1)


def build_chirps_daily(climate_dir: Path, boundaries: gpd.GeoDataFrame) -> pd.DataFrame:
    files = sorted(climate_dir.glob('chirps-v2.0.*.days_p25.nc'))
    if not files:
        raise FileNotFoundError('No CHIRPS annual NetCDF files were found in Climate.zip')
    parts = []
    for fp in files:
        m = re.search(r'chirps-v2\.0\.(\d{4})\.days_p25\.nc$', fp.name)
        if not m:
            continue
        year = int(m.group(1))
        with rasterio.open(str(fp)) as src:
            dates = pd.date_range(f'{year}-01-01', periods=src.count, freq='D')
            for _, row in boundaries.iterrows():
                arr, _ = rio_mask(src, [row.geometry.__geo_interface__], crop=True, all_touched=True, filled=False)
                vals = arr.filled(np.nan)
                vals = np.where(vals < -1000, np.nan, vals)
                mean = np.nanmean(vals, axis=(1,2))
                parts.append(pd.DataFrame({'district':row['shapeName'], 'date':dates, 'rain_mm':mean}))
    return pd.concat(parts, ignore_index=True)


def idw_power_to_district(power_df: pd.DataFrame, param: str, boundaries: gpd.GeoDataFrame, k: int = 4) -> pd.DataFrame:
    points = power_df[['LAT','LON']].drop_duplicates().reset_index(drop=True)
    points_gdf = gpd.GeoDataFrame(
        points.copy(), geometry=gpd.points_from_xy(points['LON'], points['LAT']), crs='EPSG:4326'
    ).to_crs('EPSG:32736')
    bproj = boundaries.to_crs('EPSG:32736')
    centroids = bproj.geometry.centroid
    tree = cKDTree(np.column_stack([points_gdf.geometry.x, points_gdf.geometry.y]))

    p = power_df.merge(points.reset_index().rename(columns={'index':'pt_idx'}), on=['LAT','LON'], how='left')
    mat = p.pivot(index='date', columns='pt_idx', values=param).reindex(columns=range(len(points)))
    parts = []
    for idx, row in boundaries.iterrows():
        c = centroids.iloc[idx]
        dist, inds = tree.query([c.x, c.y], k=min(k, len(points)))
        dist = np.atleast_1d(dist)
        inds = np.atleast_1d(inds)
        weights = 1.0 / np.maximum(dist, 1.0) ** 2
        weights = weights / weights.sum()
        vals = mat.iloc[:, inds].to_numpy(dtype=float)
        series = np.nansum(vals * weights, axis=1)
        parts.append(pd.DataFrame({'district':row['shapeName'], 'date':mat.index, param:series}))
    return pd.concat(parts, ignore_index=True)


def build_power_daily(climate_dir: Path, boundaries: gpd.GeoDataFrame) -> pd.DataFrame:
    files = sorted(climate_dir.glob('POWER*.csv'))
    if not files:
        raise FileNotFoundError('No NASA POWER CSV files were found in Climate.zip')
    params = {}
    for fp in files:
        d = read_power_csv(fp)
        param_cols = [c for c in d.columns if c not in ['LAT','LON','YEAR','DOY']]
        if len(param_cols) != 1:
            continue
        param = param_cols[0]
        d[param] = pd.to_numeric(d[param], errors='coerce').replace(-999, np.nan)
        d['date'] = pd.to_datetime(d['YEAR'].astype(str), format='%Y') + pd.to_timedelta(d['DOY'] - 1, unit='D')
        params.setdefault(param, []).append(d[['LAT','LON','date',param]])

    required = ['RH2M','T2M']
    for p in required:
        if p not in params:
            raise ValueError(f'NASA POWER variable {p} not found.')

    rh = pd.concat(params['RH2M'], ignore_index=True).drop_duplicates(['LAT','LON','date'])
    temp = pd.concat(params['T2M'], ignore_index=True).drop_duplicates(['LAT','LON','date'])
    rh_d = idw_power_to_district(rh, 'RH2M', boundaries)
    t_d = idw_power_to_district(temp, 'T2M', boundaries)
    return rh_d.merge(t_d, on=['district','date'], how='outer', validate='one_to_one')


def build_climate_features(climate_dir: Path, output_dir: Path, qa: dict) -> pd.DataFrame:
    nested_zip = climate_dir / 'geoBoundaries-RWA-ADM2-all.zip'
    geo_dir = output_dir / '_geo_temp'
    if geo_dir.exists():
        shutil.rmtree(geo_dir)
    extract_zip(nested_zip, geo_dir)
    geojson = next(geo_dir.glob('*ADM2.geojson'))
    boundaries = gpd.read_file(geojson)
    if len(boundaries) != 30:
        raise ValueError(f'Expected 30 Rwanda districts; boundary file contains {len(boundaries)}.')
    qa['boundary_district_count'] = len(boundaries)
    qa['boundary_districts'] = sorted(boundaries['shapeName'].tolist())

    rain = build_chirps_daily(climate_dir, boundaries)
    power = build_power_daily(climate_dir, boundaries)
    daily = rain.merge(power, on=['district','date'], how='outer', validate='one_to_one').sort_values(['district','date'])
    qa['climate_daily_rows'] = len(daily)
    qa['climate_date_min'] = str(daily['date'].min().date())
    qa['climate_date_max'] = str(daily['date'].max().date())
    qa['climate_missing_rain'] = int(daily['rain_mm'].isna().sum())
    qa['climate_missing_humidity'] = int(daily['RH2M'].isna().sum())
    qa['climate_missing_temperature'] = int(daily['T2M'].isna().sum())

    # Window features aligned to every possible harvest date in daily table.
    parts = []
    for district, d in daily.groupby('district', sort=False):
        d = d.sort_values('date').copy()
        rain_s = d['rain_mm']
        d['rain_30d_before_mm'] = rain_s.shift(1).rolling(30, min_periods=30).sum()
        d['rain_14d_after_mm'] = rain_s.iloc[::-1].rolling(14, min_periods=14).sum().iloc[::-1]
        rainy = (rain_s >= 1.0).astype(float)
        d['rainy_days_14d_after'] = rainy.iloc[::-1].rolling(14, min_periods=14).sum().iloc[::-1]
        d['humidity_mean_7d_before_pct'] = d['RH2M'].shift(1).rolling(7, min_periods=7).mean()
        d['humidity_mean_14d_after_pct'] = d['RH2M'].iloc[::-1].rolling(14, min_periods=14).mean().iloc[::-1]
        d['temperature_mean_14d_after_c'] = d['T2M'].iloc[::-1].rolling(14, min_periods=14).mean().iloc[::-1]
        parts.append(d)
    lookup = pd.concat(parts, ignore_index=True)
    daily.to_csv(output_dir / 'district_daily_climate_features.csv', index=False)
    shutil.rmtree(geo_dir, ignore_errors=True)
    return lookup


def estimate_harvest_date(year: int, season: str, period) -> tuple[pd.Timestamp, str]:
    if pd.isna(period):
        month, day = SEASON_FALLBACK.get(season, (6,30))
        method = 'season_fallback_missing_period'
    elif period == 'Other season (applicable for all crops)':
        month, day = SEASON_FALLBACK.get(season, (6,30))
        method = 'season_fallback_other_season'
    elif period in HARVEST_MIDPOINTS:
        month, day = HARVEST_MIDPOINTS[period]
        method = 'period_midpoint'
    else:
        month, day = SEASON_FALLBACK.get(season, (6,30))
        method = 'season_fallback_unparsed'
    y = int(year)
    # NISR Season A is named for the ending year; Nov/Dec harvest windows belong to prior calendar year.
    if season == 'A' and month in (11,12):
        y -= 1
    return pd.Timestamp(y, month, day), method


def attach_climate(df: pd.DataFrame, climate_lookup: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    est = [estimate_harvest_date(y, s, p) for y, s, p in zip(out['year'], out['season'], out['expected_harvest_period'])]
    out['harvest_date_est'] = [x[0] for x in est]
    out['harvest_date_method'] = [x[1] for x in est]
    cols = [
        'district','date','rain_30d_before_mm','rain_14d_after_mm','rainy_days_14d_after',
        'humidity_mean_7d_before_pct','humidity_mean_14d_after_pct','temperature_mean_14d_after_c'
    ]
    out = out.merge(climate_lookup[cols], left_on=['district','harvest_date_est'], right_on=['district','date'], how='left', validate='many_to_one')
    out = out.drop(columns='date')
    weather_cols = [
        'rain_30d_before_mm','rain_14d_after_mm','rainy_days_14d_after',
        'humidity_mean_7d_before_pct','humidity_mean_14d_after_pct','temperature_mean_14d_after_c'
    ]
    out['climate_window_complete'] = out[weather_cols].notna().all(axis=1).astype('int8')
    return out


# ----------------------------- Final schema / dictionary -----------------------------

def build_data_dictionary() -> pd.DataFrame:
    rows = [
        ('farmer_crop_obs_id','identifier','Unique row identifier; never use as ML feature.'),
        ('year','split/context','Survey year. Recommended temporal validation: train earlier year, test later year.'),
        ('season','candidate_preharvest_feature','NISR agricultural season A/B/C.'),
        ('province','candidate_preharvest_feature','Province.'),
        ('district','candidate_preharvest_feature','District; also final map aggregation unit.'),
        ('segment_id','identifier','Survey segment identifier; retain for traceability, not ML feature.'),
        ('farmer_id','identifier','Farmer identifier within survey context; not ML feature.'),
        ('farmer_type','candidate_preharvest_feature','Farmer type.'),
        ('farmer_gender','candidate_preharvest_feature','Farmer gender.'),
        ('farmer_age','candidate_preharvest_feature','Farmer age; values >100 set missing.'),
        ('crop_name','candidate_preharvest_feature','Standardized crop name.'),
        ('crop_category','candidate_preharvest_feature','Crop category, harmonized across years where possible.'),
        ('expected_harvest_period','candidate_preharvest_feature','Survey harvest-period category.'),
        ('harvest_date_est','derived_context','Approximate midpoint used only to align climate windows.'),
        ('harvest_date_method','quality/context','Whether midpoint came from survey period or season fallback.'),
        ('seed_type_sown','candidate_preharvest_feature','Seed type.'),
        ('n_plots_crop','candidate_preharvest_feature','Number of plots on which farmer reports this crop.'),
        ('crop_plot_area_ha_total','candidate_preharvest_feature','Sum of areas of plots containing crop; not exact crop area under intercropping.'),
        ('num_crops_in_plot','candidate_preharvest_feature','Mean number of crops on crop-bearing plots.'),
        ('used_improved_seed_share_yes','candidate_preharvest_feature','Share of crop rows reporting improved seed.'),
        ('used_organic_fert_share_yes','candidate_preharvest_feature','Share of crop plots reporting organic fertilizer.'),
        ('used_inorganic_fert_share_yes','candidate_preharvest_feature','Share of crop plots reporting inorganic fertilizer.'),
        ('used_pesticide_share_yes','candidate_preharvest_feature','Share of crop plots reporting pesticide/fungicide.'),
        ('used_mech_equipment_share_yes','candidate_preharvest_feature','Share of crop plots reporting mechanical equipment.'),
        ('plot_irrigated_this_season_share_yes','candidate_preharvest_feature','Share of crop plots reported irrigated.'),
        ('erosion_degree','candidate_preharvest_feature','Reported erosion class; likely secondary feature.'),
        ('irrigation_technique','candidate_preharvest_feature','Irrigation technique where applicable.'),
        ('rain_30d_before_mm','candidate_weather_feature','District CHIRPS rainfall in 30 days before estimated harvest.'),
        ('rain_14d_after_mm','weather_requires_forecast_in_live_app','Historical 14-day post-harvest rainfall; replace with forecast in live prediction.'),
        ('rainy_days_14d_after','weather_requires_forecast_in_live_app','Count of >=1mm rainy days after estimated harvest; use forecast operationally.'),
        ('humidity_mean_7d_before_pct','candidate_weather_feature','NASA POWER district-interpolated humidity before harvest.'),
        ('humidity_mean_14d_after_pct','weather_requires_forecast_in_live_app','Historical post-harvest humidity; replace with forecast operationally.'),
        ('temperature_mean_14d_after_c','weather_requires_forecast_in_live_app','Historical post-harvest temperature; replace with forecast operationally.'),
        ('climate_window_complete','quality_flag','1 only when all weather windows are fully covered by available climate data.'),
        ('total_production_qty','exposure_not_preharvest_feature','Reported crop production. Useful for district burden/prioritization, not a pre-harvest risk predictor.'),
        ('total_harvest_kg','exposure_not_preharvest_feature','Sum of plot harvest quantities; descriptive/exposure.'),
        ('storage_type','context_use_carefully','Observed storage arrangement; do not use blindly for a before-harvest forecast.'),
        ('qty_stored_future_use_kg','postoutcome_context','Post-harvest disposition; not a predictive feature.'),
        ('loss_storage_kg','target_component','Main target component.'),
        ('loss_processing_kg','target_component','Main target component.'),
        ('loss_packaging_kg','target_component','Main target component.'),
        ('actionable_loss_kg','target','Storage + processing + packaging losses.'),
        ('actionable_loss_any','target','Binary target: any actionable handling loss.'),
        ('actionable_loss_rate_pct','target_severity','Actionable loss as % of reported total crop production.'),
        ('loss_transport_kg','descriptive_only','Shown historically, excluded from main ML target due lack of transport-condition predictors.'),
        ('loss_sales_kg','descriptive_only','Historical loss at sales; not part of actionable handling target.'),
        ('reported_postharvest_loss_kg_all_stages','descriptive_only','Transport + storage + processing + packaging + sales.'),
        ('actionable_target_invalid_negative_source','quality_flag','Any negative storage/processing/packaging source value; excluded from regression target.'),
        ('actionable_target_invalid_source_inconsistency','quality_flag','Repeated plot rows disagree on crop production or actionable loss; excluded from regression target.'),
        ('actionable_target_invalid_exceeds_production','quality_flag','1 when reported actionable loss exceeded reported production; target then set missing.'),
        ('loss_storage_kg_was_negative','quality_flag','1 when raw storage loss was negative; invalid value set missing.'),
        ('survey_weight_mean','survey_design','Mean plot survey weight; do not use as ML predictor.'),
    ]
    return pd.DataFrame(rows, columns=['column','role','description'])


def choose_final_columns(df: pd.DataFrame) -> pd.DataFrame:
    cols = [
        'farmer_crop_obs_id','year','season','province','district','segment_id','farmer_id',
        'farmer_type','farmer_gender','farmer_age','crop_name','crop_category','expected_harvest_period',
        'harvest_date_est','harvest_date_method','seed_type_sown','n_plots_crop','crop_plot_area_ha_total',
        'num_crops_in_plot','used_improved_seed_share_yes','used_organic_fert_share_yes',
        'used_inorganic_fert_share_yes','used_micronutrients_share_yes','used_pesticide_share_yes',
        'anti_erosion_share_yes','used_mech_equipment_share_yes','plot_irrigated_this_season_share_yes',
        'erosion_degree','irrigation_technique','rain_30d_before_mm','rain_14d_after_mm',
        'rainy_days_14d_after','humidity_mean_7d_before_pct','humidity_mean_14d_after_pct',
        'temperature_mean_14d_after_c','climate_window_complete','total_production_qty','total_harvest_kg',
        'storage_type','qty_stored_future_use_kg','loss_storage_kg','loss_processing_kg','loss_packaging_kg',
        'actionable_loss_kg','actionable_loss_any','actionable_loss_rate_pct','loss_transport_kg','loss_sales_kg',
        'reported_postharvest_loss_kg_all_stages','actionable_target_invalid_negative_source',
        'actionable_target_invalid_source_inconsistency','actionable_target_invalid_exceeds_production',
        'loss_storage_kg_was_negative','farmer_age_over100_flag','farmer_age_under15_flag','survey_weight_mean',
        'total_production_qty_source_nunique','loss_storage_kg_source_nunique',
        'loss_processing_kg_source_nunique','loss_packaging_kg_source_nunique','loss_transport_kg_source_nunique',
    ]
    return df[[c for c in cols if c in df.columns]].copy()



# ----------------------------- Final regression model table -----------------------------

MODEL_FEATURES = [
    # Categorical context available before actionable post-harvest losses occur.
    'season',
    'district',
    'crop_name',
    'farmer_type',
    # Timing / farm scale.
    'harvest_month',
    'n_plots_crop',
    'crop_plot_area_ha_total',
    'num_crops_in_plot',
    # Climate around harvest. Historical post-harvest values must be replaced by forecasts
    # when the live application scores a coming harvest period.
    'rain_30d_before_mm',
    'rain_14d_after_mm',
    'rainy_days_14d_after',
    'humidity_mean_7d_before_pct',
    'humidity_mean_14d_after_pct',
    'temperature_mean_14d_after_c',
]

MODEL_TARGET = 'actionable_loss_rate_pct'
MODEL_SPLIT_COLUMN = 'year'


def build_regression_model_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return only the columns required for the agreed regression model.

    year is retained solely for temporal splitting (train 2024 -> test 2025);
    it must not be fed to the estimator as a predictive feature.

    Target = percentage of crop production reported lost during storage + processing + packaging.
    Transport and sales losses are deliberately excluded from both X and y.
    """
    out = df.copy()
    out['harvest_month'] = pd.to_datetime(out['harvest_date_est'], errors='coerce').dt.month.astype('Int64')

    cols = [MODEL_SPLIT_COLUMN] + MODEL_FEATURES + [MODEL_TARGET]
    missing_cols = [c for c in cols if c not in out.columns]
    if missing_cols:
        raise ValueError(f'Model-ready columns are missing from the cleaned dataset: {missing_cols}')

    model = out[cols].copy()

    # A row with no valid continuous target cannot participate in regression training/evaluation.
    model = model[model[MODEL_TARGET].notna()].copy()

    # All selected predictors should be present after the completed climate pipeline. Fail loudly
    # instead of silently imputing scientifically meaningful values at this stage.
    predictor_missing = model[MODEL_FEATURES].isna().sum()
    bad = predictor_missing[predictor_missing > 0]
    if len(bad):
        raise ValueError('Selected model predictors still contain missing values: ' + bad.to_dict().__repr__())

    # Physical target constraints already handled upstream; assert them again on the final training table.
    if (model[MODEL_TARGET] < 0).any() or (model[MODEL_TARGET] > 100).any():
        raise ValueError('Regression target contains values outside the physical 0-100% range.')

    model = model.sort_values([MODEL_SPLIT_COLUMN, 'season', 'district', 'crop_name']).reset_index(drop=True)
    return model


def build_model_dictionary() -> pd.DataFrame:
    rows = [
        ('year','split_only','Calendar/survey year. Use for temporal split (2024 train, 2025 test); do NOT include in X.'),
        ('season','X_categorical','NISR agricultural season; captures seasonal crop/handling context.'),
        ('district','X_categorical','District whose persistent local post-harvest context is being scored.'),
        ('crop_name','X_categorical','Crop identity; crops differ strongly in drying, processing, packaging and storage behavior.'),
        ('farmer_type','X_categorical','Farm/organization scale type; proxy for operating/storage/processing capacity.'),
        ('harvest_month','X_numeric_or_categorical','Month of estimated harvest; captures timing beyond broad season labels.'),
        ('n_plots_crop','X_numeric','Number of plots on which the farmer grows the crop; handling/fragmentation scale.'),
        ('crop_plot_area_ha_total','X_numeric','Total sampled plot area for this farmer-crop; pre-loss scale/exposure proxy.'),
        ('num_crops_in_plot','X_numeric','Mean number of crops on relevant plots; proxy for intercropping/handling complexity.'),
        ('rain_30d_before_mm','X_numeric','CHIRPS rainfall in 30 days before estimated harvest.'),
        ('rain_14d_after_mm','X_numeric_forecast_at_deployment','CHIRPS rainfall in 14 days after harvest historically; use forecast in live scoring.'),
        ('rainy_days_14d_after','X_numeric_forecast_at_deployment','Number of >=1mm rainy days after harvest historically; use forecast in live scoring.'),
        ('humidity_mean_7d_before_pct','X_numeric','NASA POWER mean relative humidity in 7 days before harvest.'),
        ('humidity_mean_14d_after_pct','X_numeric_forecast_at_deployment','NASA POWER mean relative humidity after harvest historically; use forecast in live scoring.'),
        ('temperature_mean_14d_after_c','X_numeric_forecast_at_deployment','NASA POWER mean temperature after harvest historically; use forecast in live scoring.'),
        ('actionable_loss_rate_pct','y_regression','Continuous target: (storage + processing + packaging loss) / total crop production * 100.'),
    ]
    return pd.DataFrame(rows, columns=['column','role','description'])

def write_quality_report(path: Path, qa: dict, final: pd.DataFrame) -> None:
    lines = []
    lines.append('UNIFIED POST-HARVEST DATASET - QUALITY REPORT')
    lines.append('=' * 60)
    lines.append('')
    lines.append('SOURCE / MERGE CHECKS')
    for k in sorted(qa):
        if k == 'boundary_districts':
            continue
        lines.append(f'- {k}: {qa[k]}')
    lines.append('')
    lines.append('FINAL DATASET CHECKS')
    lines.append(f'- rows: {len(final):,}')
    lines.append(f'- columns: {len(final.columns)}')
    lines.append(f'- duplicate farmer-crop observation IDs: {int(final["farmer_crop_obs_id"].duplicated().sum())}')
    lines.append(f'- districts: {final["district"].nunique()}')
    lines.append(f'- district names: {", ".join(sorted(final["district"].dropna().unique()))}')
    lines.append('')
    lines.append('ROWS BY YEAR / SEASON')
    for (year, season), n in final.groupby(['year','season']).size().items():
        lines.append(f'- {year} Season {season}: {n:,}')
    lines.append('')
    lines.append('TARGET QUALITY')
    lines.append(f'- complete actionable binary targets: {int(final["actionable_loss_any"].notna().sum()):,}')
    lines.append(f'- positive actionable-loss cases: {int((final["actionable_loss_any"] == 1).sum()):,}')
    lines.append(f'- positive storage-loss cases: {int((final["loss_storage_kg"] > 0).sum()):,}')
    lines.append(f'- positive processing-loss cases: {int((final["loss_processing_kg"] > 0).sum()):,}')
    lines.append(f'- positive packaging-loss cases: {int((final["loss_packaging_kg"] > 0).sum()):,}')
    if 'actionable_target_invalid_negative_source' in final:
        lines.append(f'- target groups excluded for any negative actionable-loss source value: {int(final["actionable_target_invalid_negative_source"].sum()):,}')
    if 'actionable_target_invalid_source_inconsistency' in final:
        lines.append(f'- target groups excluded for inconsistent repeated production/loss values: {int(final["actionable_target_invalid_source_inconsistency"].sum()):,}')
    lines.append(f'- impossible actionable targets removed (> production): {int(final["actionable_target_invalid_exceeds_production"].sum()):,}')
    if 'loss_storage_kg_was_negative' in final:
        lines.append(f'- rows with negative raw storage loss removed: {int(final["loss_storage_kg_was_negative"].sum()):,}')
    lines.append('')
    lines.append('CLIMATE COVERAGE')
    cov = final.groupby(['year','season'])['climate_window_complete'].agg(['sum','count','mean'])
    for idx, r in cov.iterrows():
        lines.append(f'- {idx[0]} Season {idx[1]}: {int(r["sum"]):,}/{int(r["count"]):,} complete ({r["mean"]*100:.1f}%)')
    lines.append('')
    lines.append('CLIMATE WINDOW NOTE')
    climate_min = qa.get('climate_date_min', 'unknown')
    climate_max = qa.get('climate_date_max', 'unknown')
    complete_share = float(final['climate_window_complete'].mean()) if len(final) else float('nan')
    lines.append(f'- available district climate span: {climate_min} to {climate_max}.')
    lines.append(f'- overall complete harvest-window climate coverage: {complete_share*100:.1f}%.')
    if complete_share < 0.999:
        lines.append('- Rows with incomplete windows are retained and flagged climate_window_complete=0; weather is never silently shifted to a wrong date.')
    else:
        lines.append('- All final observations have complete requested climate windows.')
    lines.append('')
    lines.append('INTENTIONALLY EXCLUDED MODULES / FIELDS')
    lines.append('- 2024 Screening modules: sampling/land screening information, not needed for the farmer-crop post-harvest target.')
    lines.append('- Detailed 2024 fertilizer/pesticide modules: the production file already contains core yes/no use flags; detailed costs/types would add sparsity and weak post-harvest relevance.')
    lines.append('- Free-text storage descriptions and Kinyarwanda open text: extremely sparse and not needed for the core model; avoiding them prevents artificial category fragmentation.')
    lines.append('- Post-outcome quantities sold/consumed/transformed: excluded from final model-ready schema to reduce leakage.')
    lines.append('')
    lines.append('RECOMMENDED MODELING RULE')
    lines.append('- Do not random-split rows as the only validation. Prefer temporal validation (e.g., train on 2024 and evaluate on 2025) and compare against a simpler baseline.')
    lines.append('- Main ML task is regression on actionable_loss_rate_pct only. Binary loss labels are QA/EDA only and are not included in the model-ready CSV.')
    path.write_text('\n'.join(lines), encoding='utf-8')


# ----------------------------- Main -----------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-dir', type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument('--output-dir', type=Path, default=None)
    args = parser.parse_args()
    input_dir = args.input_dir.resolve()
    output_dir = (args.output_dir or (input_dir / 'postharvest_pipeline_output')).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    required = ['Dataset.zip','all_seasons.zip','Climate.zip']
    missing = [f for f in required if not (input_dir / f).exists()]
    if missing:
        raise FileNotFoundError(f'Missing required input files in {input_dir}: {missing}')

    qa = {}
    with tempfile.TemporaryDirectory(prefix='postharvest_build_') as td:
        temp = Path(td)
        d24_dir = temp / 'dataset24'
        d25_dir = temp / 'dataset25'
        climate_dir = temp / 'climate'
        extract_zip(input_dir / 'Dataset.zip', d24_dir)
        extract_zip(input_dir / 'all_seasons.zip', d25_dir)
        extract_zip(input_dir / 'Climate.zip', climate_dir)

        dataset_root = next((p for p in d24_dir.rglob('Dataset') if (p / 'Season A').exists()), None)
        if dataset_root is None:
            dataset_root = next(p.parent for p in d24_dir.rglob('Season A') if p.is_dir())
        user_2025 = next(d25_dir.rglob('all_seasons.csv'))
        climate_root = next((p for p in climate_dir.rglob('Climate') if p.is_dir()), climate_dir)

        # Add any standalone climate-year files supplied beside the ZIP. This makes the pipeline
        # reproducible when an extra year (e.g. 2023) is needed for harvest windows that cross
        # calendar years. Existing files inside Climate.zip are left untouched.
        for pattern in ('chirps-v2.0.*.days_p25.nc', 'POWER*.csv'):
            for fp in input_dir.glob(pattern):
                dest = climate_root / fp.name
                if not dest.exists():
                    shutil.copy2(fp, dest)

        # EDA + merge 2024 raw production/practice season by season.
        plot24_parts = [prepare_2024_season(dataset_root, s, qa) for s in ['A','B','C']]
        plot24 = pd.concat(plot24_parts, ignore_index=True, sort=False)
        qa['2024_plot_crop_rows_after_merge'] = len(plot24)

        # Re-clean user's validated 2025 merged file; validate raw B merge independently when available.
        plot25 = prepare_2025_from_user_merged(user_2025, qa)
        validate_2025_raw_b(input_dir, qa)
        qa['2025_plot_crop_rows_after_cleaning'] = len(plot25)

        # Verify all districts after standardization.
        expected_districts = {
            'Bugesera','Burera','Gakenke','Gasabo','Gatsibo','Gicumbi','Gisagara','Huye','Kamonyi','Karongi',
            'Kayonza','Kicukiro','Kirehe','Muhanga','Musanze','Ngoma','Ngororero','Nyabihu','Nyagatare',
            'Nyamagabe','Nyamasheke','Nyanza','Nyarugenge','Nyaruguru','Rubavu','Ruhango','Rulindo','Rusizi','Rutsiro','Rwamagana'
        }
        observed = set(pd.concat([plot24['district'], plot25['district']]).dropna().unique())
        if observed != expected_districts:
            raise ValueError(f'District harmonization failed. Missing={expected_districts-observed}, unexpected={observed-expected_districts}')

        # Collapse from plot/crop to farmer/crop/season/year.
        fc24 = collapse_farmer_crop(plot24)
        fc25 = collapse_farmer_crop(plot25)
        qa['2024_farmer_crop_rows'] = len(fc24)
        qa['2025_farmer_crop_rows'] = len(fc25)
        combined = pd.concat([fc24, fc25], ignore_index=True, sort=False)

        # Harmonize crop category. 2025 Season A lacks it; infer only from the same standardized crop name elsewhere.
        known = combined.dropna(subset=['crop_category'])
        crop_category_lookup = known.groupby('crop_name')['crop_category'].agg(lambda x: x.value_counts().index[0]).to_dict()
        combined['crop_category'] = combined['crop_category'].fillna(combined['crop_name'].map(crop_category_lookup))
        combined.loc[(combined['crop_name'] == 'Sugar beet') & combined['crop_category'].isna(), 'crop_category'] = 'Vegetables'

        # Target cleaning.
        combined = clean_model_targets(combined)

        # Climate extraction and matching.
        climate_lookup = build_climate_features(climate_root, output_dir, qa)
        combined = attach_climate(combined, climate_lookup)

        # Keep the richer cleaned table only in memory for QA. The file handed to modeling
        # contains ONLY split metadata + X features + the continuous regression target.
        audit = choose_final_columns(combined)
        audit = audit.sort_values(['year','season','district','segment_id','farmer_id','crop_name']).reset_index(drop=True)

        # Final grain / geography validation before trimming model columns.
        if audit['farmer_crop_obs_id'].duplicated().any():
            raise ValueError('Final grain is not unique: duplicate farmer_crop_obs_id values remain.')
        if audit['district'].nunique() != 30:
            raise ValueError('Final dataset does not contain exactly 30 standardized districts.')

        model_ready = build_regression_model_dataset(audit)
        qa['regression_rows_with_valid_target'] = len(model_ready)
        qa['regression_feature_count'] = len(MODEL_FEATURES)
        qa['regression_target'] = MODEL_TARGET
        qa['train_2024_rows'] = int((model_ready['year'] == 2024).sum())
        qa['test_2025_rows'] = int((model_ready['year'] == 2025).sum())

        out_csv = output_dir / 'postharvest_regression_model_ready_2024_2025.csv'
        model_ready.to_csv(out_csv, index=False)
        build_model_dictionary().to_csv(output_dir / 'postharvest_regression_model_dictionary.csv', index=False)
        manifest = {
            'split_column_not_in_X': MODEL_SPLIT_COLUMN,
            'recommended_temporal_split': {'train': 'year == 2024', 'test': 'year == 2025'},
            'features_X': MODEL_FEATURES,
            'target_y': MODEL_TARGET,
            'excluded_from_target': ['transport loss', 'sales loss'],
            'excluded_from_X': [
                'IDs and survey QA flags', 'binary loss indicators', 'all loss component columns',
                'post-harvest quantities/disposition fields', 'storage_type (96% missing and not reliably known before scoring)',
                'province/crop_category redundant with district/crop_name', 'farmer age/gender demographics',
                'fertilizer/seed/irrigation/erosion variables with weak direct post-harvest relevance',
                'actual production quantities reserved for later burden/prioritization rather than intrinsic loss-rate prediction'
            ],
        }
        (output_dir / 'postharvest_regression_model_manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
        write_quality_report(output_dir / 'unified_postharvest_quality_report.txt', qa, audit)
        (output_dir / 'quality_metrics.json').write_text(json.dumps(qa, indent=2, default=str), encoding='utf-8')

        print(f'Created: {out_csv}')
        print(f'Model rows: {len(model_ready):,} | Columns: {len(model_ready.columns)}')
        print(f'X features: {len(MODEL_FEATURES)} | y: {MODEL_TARGET}')
        print(f'Temporal split rows -> train 2024: {(model_ready.year == 2024).sum():,} | test 2025: {(model_ready.year == 2025).sum():,}')
        print('No missing values in selected X or y:', int(model_ready[MODEL_FEATURES + [MODEL_TARGET]].isna().sum().sum()) == 0)


if __name__ == '__main__':
    main()
