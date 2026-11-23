from pathlib import Path
import pandas as pd
import numpy as np
P=Path(__file__).resolve().parents[1]/'results/revision_20260925'
f=pd.read_csv(P/'flexible_trend_families.csv')
print('POINT FLEXIBLE CHECK')
print(f[f.origin.eq('representative_point')][['study','relation','trend','outcome','peak_threshold','omnibus_q']].to_string(index=False))
print('FLEXIBLE CHECK COUNTS')
print(f.groupby(['trend','outcome']).agg(n=('omnibus_q','size'),selected=('omnibus_q',lambda v:(v<=.05).sum()),min_q=('omnibus_q','min')).to_string())
print('ALL SELECTED SPLINE FAMILIES')
print(f[f.trend.ne('piecewise_linear')&f.omnibus_q.le(.05)].to_string(index=False))
r=pd.read_csv(P/'random_postcode_benchmark.csv')
print('RANDOM BENCHMARK')
print(r[r.threshold.eq(15)&r.benchmark.eq('weighted')][['study','relation','draw','mean_flip_percent','p025_flip_percent','p975_flip_percent']].to_string(index=False))
print('NULL CALIBRATION')
print(pd.read_csv(P/'independent_null_calibration.csv')[['spec','mean_panel_detection','probability_any_panel']].to_string(index=False))
print('ANALYTIC')
print(pd.read_csv(P/'analytic_signal_projection.csv')[['study','relation','spec','conditional_signal_coefficient']].to_string(index=False))
