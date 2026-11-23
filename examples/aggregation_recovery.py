"""Synthetic illustration of aggregating an origin-level step to neighbourhoods."""
import numpy as np

def main():
    rng=np.random.default_rng(20260903)
    n_areas,n_origins=400,100
    point_time=rng.uniform(5,25,n_areas)
    origin_times=point_time[:,None]+rng.normal(0,5,(n_areas,n_origins))
    outside_share=(origin_times>15).mean(axis=1)
    point_step=(point_time>15).astype(float)
    nuisance=np.column_stack([np.ones(n_areas),point_time/5])
    delta=1.5
    background=nuisance@np.array([2.,.3])+rng.normal(0,.1,n_areas)
    injected=background+delta*outside_share
    point_design=np.column_stack([nuisance,point_step])
    share_design=np.column_stack([nuisance,outside_share])
    paired=lambda x:np.linalg.lstsq(x,injected,rcond=None)[0][-1]-np.linalg.lstsq(x,background,rcond=None)[0][-1]
    point_change,share_change=paired(point_design),paired(share_design)
    d=point_step-nuisance@np.linalg.lstsq(nuisance,point_step,rcond=None)[0]
    a=outside_share-nuisance@np.linalg.lstsq(nuisance,outside_share,rcond=None)[0]
    expected_point=delta*(d@a)/(d@d)
    np.testing.assert_allclose(point_change,expected_point,atol=1e-10)
    np.testing.assert_allclose(share_change,delta,atol=1e-10)
    print('Synthetic example; these are not the empirical paper estimates.')
    print(f'Injected change: {delta:.3f}; point-fit coefficient change: {point_change:.3f}; share-fit coefficient change: {share_change:.3f}')

if __name__=='__main__':
    main()
