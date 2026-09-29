#!/usr/bin/env python3
"""Seven-year, outcome-blind leave-one-year-out pilot of v3c weather generation."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from audit_v22_weather_v3_development import calendar_path, features
from slowlab.v22_private_weather import read_source_year
from slowlab.v22_private_weather_v3c import BlockWeatherGeneratorV3c
from slowlab.v22_weather_similarity import audit_nonreplay


YEARS = (2013, 2014, 2016, 2017, 2018, 2019, 2020)


def fair_energy_score(rows, truth):
    """Use off-diagonal pairs so unequal ensemble sizes do not bias comparison."""
    rows = np.asarray(rows, dtype=np.float64)
    if len(rows) < 2:
        return None
    first = np.linalg.norm(rows - truth, axis=1).mean()
    pairwise = np.linalg.norm(rows[:, None, :] - rows[None, :, :], axis=2)
    second = pairwise.sum() / (2 * len(rows) * (len(rows) - 1))
    return float(first - second)


def read_years(archive: Path, metadata_path: Path, cache: Path):
    metadata = json.loads(metadata_path.read_text())
    if (set(metadata['years']) != {'2013', '2014', '2016'}
            or set(metadata['excluded_years']) != {'2012'}
            or hashlib.sha256(archive.read_bytes()).hexdigest() != metadata['archive_sha256']):
        raise ValueError('new development weather identity changed')
    with np.load(archive, allow_pickle=False) as data:
        if set(data.files) != {'2013', '2014', '2016'}:
            raise ValueError('new development weather archive years changed')
        years = {year: np.asarray(data[str(year)], dtype=np.float64)
                 for year in (2013, 2014, 2016)}
    for year, values in years.items():
        digest = hashlib.sha256(values.astype('<f8').tobytes()).hexdigest()
        if digest != metadata['years'][str(year)]['sha256_f64_le']:
            raise ValueError('new development weather matrix hash changed')
    plan = ROOT / 'configs/v22_weather_gapfilled_plan.json'
    years.update({year: read_source_year(cache, plan, year)
                  for year in (2017, 2018, 2019, 2020)})
    return years, metadata


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cache', type=Path, required=True)
    parser.add_argument('--new-years', type=Path, required=True)
    parser.add_argument('--new-metadata', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--candidates-per-fold', type=int, default=12)
    args = parser.parse_args()
    if not 1 <= args.candidates_per_fold <= 32:
        raise ValueError('bounded development pilot requires 1-32 candidates per fold')
    started = time.monotonic()
    years, metadata = read_years(args.new_years, args.new_metadata, args.cache)
    if tuple(sorted(years)) != YEARS:
        raise ValueError('development year set changed')
    folds = []
    for heldout in YEARS:
        training = {year: years[year] for year in YEARS if year != heldout}
        generator = BlockWeatherGeneratorV3c(training)
        truth = features(calendar_path(years[heldout], heldout, np), generator.scale, np)
        baseline = [features(calendar_path(path, year, np), generator.scale, np)
                    for year, path in training.items()]
        entries, generated = [], []
        for offset in range(args.candidates_per_fold):
            seed = 20262000 + heldout * 100 + offset
            entry = {'development_seed': seed}
            try:
                candidate = generator.sample_candidate(seed)
            except ValueError as exc:
                entry.update(physical_support=False, nonreplay=None,
                             support_error=str(exc))
                entries.append(entry)
                continue
            entry['physical_support'] = True
            check = audit_nonreplay(candidate, training, generator.scale, threshold=.08)
            entry['nonreplay'] = check['passed']
            entry['minimum_24h_distance'] = check['by_window']['24h']['minimum_standardized_rmse']
            entry['minimum_6h_distance'] = check['by_window']['6h']['minimum_standardized_rmse']
            if check['passed']:
                summary = features(candidate, generator.scale, np)
                entry['wind_daily_variance_ratio_vs_heldout'] = float(
                    summary['daily_variance'][2] / truth['daily_variance'][2])
                generated.append(summary)
            entries.append(entry)
        fold = {'heldout_year': heldout, 'fit_years': list(training), 'rows': entries,
                'physical_support_count': sum(e['physical_support'] for e in entries),
                'nonreplay_count': len(generated)}
        if generated:
            matrices = np.stack([row['monthly'] for row in generated])
            low, high = np.quantile(matrices, [.05, .95], axis=0)
            fold['monthly_5_95_coverage'] = float(np.mean(
                (truth['monthly'] >= low) & (truth['monthly'] <= high)))
            fold['weather_fair_energy_score'] = fair_energy_score(
                [row['vector'] for row in generated], truth['vector'])
            fold['whole_year_resample_baseline_fair_energy_score'] = fair_energy_score(
                [row['vector'] for row in baseline], truth['vector'])
            ratios = [e['wind_daily_variance_ratio_vs_heldout']
                      for e in entries if e['nonreplay']]
            fold['generated_wind_daily_variance_ratio_range'] = [min(ratios), max(ratios)]
        folds.append(fold)
    report = {
        'scope': 'seven same-station development weather years; no formal sites, crop outcomes, or 2015/2025 holdout',
        'candidate_version': 'block-weather-v3c-expanded-development',
        'year_set': list(YEARS),
        'excluded_years': metadata['excluded_years'],
        'new_years_archive_sha256': metadata['archive_sha256'],
        'new_years_metadata_sha256': hashlib.sha256(args.new_metadata.read_bytes()).hexdigest(),
        'legacy_weather_plan_sha256': hashlib.sha256(
            (ROOT / 'configs/v22_weather_gapfilled_plan.json').read_bytes()).hexdigest(),
        'legacy_weather_manifest_sha256': hashlib.sha256(
            (args.cache / 'manifest.json').read_bytes()).hexdigest(),
        'candidate_code_sha256': hashlib.sha256(
            (ROOT / 'slowlab/v22_private_weather_v3c.py').read_bytes()).hexdigest(),
        'audit_code_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'candidates_per_fold': args.candidates_per_fold,
        'nonreplay_rule': 'aligned 24h and 6h windows against training years only; threshold 0.08; sliding audit pending',
        'energy_score_estimator': 'off-diagonal U-statistic; undefined for fewer than two accepted candidates',
        'folds': folds,
        'elapsed_seconds': time.monotonic() - started,
    }
    args.out.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(json.dumps({'folds': len(folds),
                      'accepted': [fold['nonreplay_count'] for fold in folds],
                      'elapsed_seconds': report['elapsed_seconds']}))


if __name__ == '__main__':
    main()
