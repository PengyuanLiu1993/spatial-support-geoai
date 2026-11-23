"""Check the published aggregate tables without external geographical inputs."""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
REF = ROOT / 'reference_results/revision_20260925'

def main():
    counts = pd.read_csv(REF / 'origin_comparator_counts.csv')
    np.testing.assert_array_equal(counts.flip_count, counts.point_uncovered_to_majority_covered + counts.point_covered_to_majority_uncovered)
    np.testing.assert_allclose(counts.flip_percent, 100 * counts.flip_count / counts.n_lsoas)
    primary = pd.read_csv(REF / 'primary_15min_comparators.csv')
    assert len(primary) == 8
    original = primary[primary.origin.eq('representative_point')].set_index(['study','relation'])
    centres = primary[primary.origin.eq('ons_population_weighted_centroid')].set_index(['study','relation']).reindex(original.index)
    assert (centres.flip_count < original.flip_count).all()
    families = pd.read_csv(REF / 'flexible_trend_families.csv')
    assert len(families) == 216 and families.omnibus_q.between(0, 1).all()
    projection = pd.read_csv(REF / 'analytic_signal_projection.csv')
    np.testing.assert_allclose(projection.loc[projection.spec.str.endswith('share_fit'), 'conditional_signal_coefficient'], 1, atol=1e-10)
    recovery = pd.read_csv(REF / 'remedy_injection_summary.csv')
    rates = recovery[recovery.absolute_step.eq(1.5)].groupby('spec').mean(numeric_only=True).mean_panel_detection
    expected = {'point_signal_point_fit':.91925, 'postcode_signal_point_fit':.119625,
                'postcode_signal_share_fit':.412375, 'population_signal_share_fit':.456375}
    for spec,value in expected.items():
        np.testing.assert_allclose(rates[spec], value, atol=0.0001)
    manifest=json.loads((ROOT/'docs/source_manifest.json').read_text(encoding='utf-8'))
    for item in manifest:
        if item['file'].startswith('reference_results/'):
            assert hashlib.sha256((ROOT/item['file']).read_bytes()).hexdigest()==item['release_sha256'],item['file']
    print('Reference tables verified: 8 primary comparisons, 216 families and matching-share recovery.')
    print((100*rates).round(3).to_string())

if __name__=='__main__':
    main()
