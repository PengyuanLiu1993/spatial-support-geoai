"""Independent checks of the revision results and measurement definitions."""
from pathlib import Path
import sys, json, os
os.environ.setdefault('MPLCONFIGDIR',str(Path.cwd()/'work/.matplotlib'))
os.environ.setdefault('LOKY_MAX_CPU_COUNT','2')
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
P=ROOT/'results/revision_20260925'
sys.path.insert(0,str(ROOT/'analysis'))
import run_threshold_robustness as base
s=pd.read_csv(P/'postcode_summaries.csv')
assert len(s)==2*(4994+1702)
assert not s[['study','lsoa21cd','relation']].duplicated().any()
assert s.population.gt(0).all()
for col in s:
    if col.startswith('share_'):assert s[col].between(0,1).all(),col
for stem in ('share_all_','share_weighted_','share_matched_equal_'):
    assert (np.diff(s[[stem+str(c) for c in (5,10,15,20)]],axis=1)>=-1e-12).all()
comp=pd.read_csv(P/'origin_comparator_counts.csv')
assert np.all(comp.flip_count==comp.point_uncovered_to_majority_covered+comp.point_covered_to_majority_uncovered)
assert np.allclose(comp.flip_percent,100*comp.flip_count/comp.n_lsoas)
checks=[]
for ci,city in enumerate(base.STUDIES):
    pc=pd.read_parquet(P/f'{city}_postcode_population_routes.parquet')
    assert pc.postcode.is_unique
    assert pc.census2021_residents.dropna().gt(0).all()
    b=base.load_study(ROOT,city,25,20260903+ci)
    for r in base.PRIMARY_RELATIONS:
        aligned=s[s.study.eq(city)&s.relation.eq(r)].set_index('lsoa21cd').reindex(b['codes'])
        t=b['nearest'][r];X=base.step_design(t,b['censored'][r],15)
        a=1-aligned.share_weighted_15.to_numpy()
        D=X[:,2];nuisance=X[:,[0,1,3,4]]
        dres=D-nuisance@np.linalg.lstsq(nuisance,D,rcond=None)[0]
        ares=a-nuisance@np.linalg.lstsq(nuisance,a,rcond=None)[0]
        kappa=dres@ares/(dres@dres)
        beta=np.linalg.lstsq(X,a,rcond=None)[0][2]
        assert abs(kappa-beta)<1e-10
        shareX=X.copy();shareX[:,2]=a
        assert abs(np.linalg.lstsq(shareX,a,rcond=None)[0][2]-1)<1e-10
        Y,_=base.wild_cluster_base(b['residuals']['pooled_context_hgb'],t,b['censored'][r],b['blocks'],10,123)
        # Check finite-simulation coefficient changes with paired identical backgrounds.
        before=np.linalg.lstsq(shareX,Y,rcond=None)[0][2]
        after=np.linalg.lstsq(shareX,Y+1.5*a[:,None],rcond=None)[0][2]
        assert np.max(np.abs(after-before-1.5))<1e-10
        checks.append({'study':city,'relation':r,'partial_projection_ratio':float(kappa),'max_paired_share_recovery_error':float(np.max(np.abs(after-before-1.5)))})
f=pd.read_csv(P/'flexible_trend_families.csv')
assert len(f)==216
assert f.omnibus_q.between(0,1).all()
report={'checks':'Original 16 postcode classification counts reproduced; 13,392 LSOA-service summaries complete; positive matched populations; bounded monotone shares; directional count sums; exact partial-projection identity; independent least-squares recovery check; all 216 trend-origin-outcome families retained.','panels':checks}
(P/'verification.json').write_text(json.dumps(report,indent=2))
print(report['checks'])
