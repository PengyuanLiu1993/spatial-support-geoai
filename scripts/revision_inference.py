"""Flexible trend checks and a directly tested coverage-share remedy."""
from pathlib import Path
import sys, json, os
os.environ.setdefault('OMP_NUM_THREADS','2')
os.environ.setdefault('OPENBLAS_NUM_THREADS','2')
import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline

ROOT=Path(__file__).resolve().parents[1]
os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'work/.matplotlib'))
sys.path.insert(0,str(ROOT/'analysis'))
import run_threshold_robustness as base
import run_tight_story_analysis as old
OUT=ROOT/'results/revision_20260925'
SEED=20260903
NBOOT=2000
NCAL=2000
NSIM=1000
THRESHOLDS=(5.,10.,15.,20.)

def natural_basis(t,df):
    knots=np.array([0,5,15,25,40] if df==4 else [0,5,10,15,20,30,40],float)
    b=CubicSpline(knots,np.eye(len(knots)),bc_type='natural')(t)
    return b[:,1:]

def make_design(t,censor,c,trend):
    if trend=='piecewise_linear':return base.step_design(t,censor,c)
    b=natural_basis(t,int(trend[-1]))
    return np.column_stack((np.ones(len(t)),b[:,0],t>c,b[:,1:],censor))

def flexible_scan(studies,summaries):
    families=[];coefficients=[];curves=[]
    for ci,city in enumerate(base.STUDIES):
        bundle=studies[city];codes=bundle['codes']
        routed=pd.read_parquet(OUT/f'{city}_routed_origins.parquet')
        for ri,r in enumerate(base.PRIMARY_RELATIONS):
            s=summaries.query('study==@city and relation==@r').set_index('lsoa21cd').reindex(codes)
            pwc=routed.query("origin_type=='ons_population_weighted_centroid'").set_index('lsoa21cd').reindex(codes)[f'{r}_minutes'].to_numpy()
            exposures={'representative_point':bundle['nearest'][r],'ons_population_weighted_centroid':np.minimum(pwc,40),'postcode_median':s.median_unweighted.to_numpy(),'postcode_p90':s.p90_unweighted.to_numpy(),'population_weighted_median':s.weighted_q50.to_numpy(),'population_weighted_p90':s.weighted_q90.to_numpy()}
            for oi,(origin,t) in enumerate(exposures.items()):
                valid=np.isfinite(t);t=np.asarray(t)[valid];censor=(t>=40-1e-9).astype(float)
                blocks=bundle['blocks'][valid]
                outcome_names,Y=old.focused_outcomes(bundle);Y=Y[valid]
                unique,counts=old.block_counts(blocks,NBOOT,SEED+ci*100000+ri*10000+oi*100)
                for trend in ('piecewise_linear','natural_spline4','natural_spline6'):
                    est=[];draws=[];candidates=[]
                    for c in THRESHOLDS:
                        if min(np.sum(t<=c),np.sum(t>c))<max(50,.04*len(t)):continue
                        X=make_design(t,censor,c,trend)
                        a,b=old.bootstrap_steps_with_counts(X,Y,blocks,unique,counts)
                        candidates.append(c);est.append(a);draws.append(b)
                    if not est:continue
                    est=np.stack(est);draws=np.stack(draws,axis=1)
                    se=np.std(draws,axis=0,ddof=1)
                    for mi,name in enumerate(outcome_names):
                        obs=est[:,mi]/se[:,mi]
                        null=np.max(np.abs((draws[:,:,mi]-est[None,:,mi])/se[None,:,mi]),axis=1)
                        p=(1+np.sum(null>=np.max(np.abs(obs))))/(NBOOT+1)
                        crit=np.quantile(null,.95);peak=np.argmax(np.abs(obs))
                        family={'study':city,'relation':r,'origin':origin,'trend':trend,'outcome':name,'n':len(t),'n_candidates':len(candidates),'peak_threshold':candidates[peak],'omnibus_p':p}
                        families.append(family)
                        for j,c in enumerate(candidates):
                            coefficients.append({**family,'threshold':c,'step':est[j,mi],'bootstrap_se':se[j,mi],'simultaneous_lower':est[j,mi]-crit*se[j,mi],'simultaneous_upper':est[j,mi]+crit*se[j,mi]})
                    if origin in ('representative_point','population_weighted_median') and trend=='natural_spline6':
                        b=natural_basis(t,6);X=np.column_stack((np.ones(len(t)),b,censor))
                        beta=np.linalg.lstsq(X,Y,rcond=None)[0]
                        grid=np.linspace(0,35,141);fit=np.column_stack((np.ones(len(grid)),natural_basis(grid,6),np.zeros(len(grid))))@beta
                        for j,x in enumerate(grid):
                            curves.append({'study':city,'relation':r,'origin':origin,'time':x,**{n:float(fit[j,i]) for i,n in enumerate(outcome_names)}})
                print('Flexible scan',city,r,origin,flush=True)
    f=pd.DataFrame(families)
    f['omnibus_q']=f.groupby(['origin','trend','outcome']).omnibus_p.transform(lambda x:base.bh_fdr(x.to_numpy()))
    co=pd.DataFrame(coefficients).merge(f[['study','relation','origin','trend','outcome','omnibus_q']],on=['study','relation','origin','trend','outcome'],validate='many_to_one')
    f.to_csv(OUT/'flexible_trend_families.csv',index=False)
    co.to_csv(OUT/'flexible_trend_coefficients.csv',index=False)
    pd.DataFrame(curves).to_csv(OUT/'smooth_trend_curves.csv',index=False)

def model_stats(X,Y,blocks):
    inverse=np.linalg.pinv(X.T@X,rcond=1e-10)
    beta=inverse@X.T@Y
    residual=Y-X@beta
    projection=X@inverse[:,2]
    unique=np.unique(blocks)
    scores=np.stack([np.sum(projection[blocks==b,None]*residual[blocks==b],axis=0) for b in unique])
    n,k=X.shape;g=len(unique)
    correction=g/(g-1)*(n-1)/(n-k)
    return beta[2],scores,correction

def share_design(t,censor,c,share):
    return np.column_stack((np.ones(len(t)),t/5,1-share,np.maximum(t-c,0)/5,censor))

def injection(studies,summaries):
    specs=('point_signal_point_fit','postcode_signal_point_fit','postcode_signal_share_fit','population_signal_point_fit','population_signal_share_fit')
    panels=[];analytic=[]
    for ci,city in enumerate(base.STUDIES):
        b=studies[city];codes=b['codes']
        for ri,r in enumerate(base.PRIMARY_RELATIONS):
            s=summaries.query('study==@city and relation==@r').set_index('lsoa21cd').reindex(codes)
            assert np.isfinite(s.share_weighted_15).all()
            t=b['nearest'][r];censor=b['censored'][r];blocks=b['blocks']
            pseudo,_=base.wild_cluster_base(b['residuals']['pooled_context_hgb'],t,censor,blocks,NCAL+NSIM,20260925+ci*1000+ri*100)
            signals={'point':(t>15).astype(float),'postcode':1-s.share_all_15.to_numpy(),'population':1-s.share_weighted_15.to_numpy()}
            panel={'study':city,'relation':r,'specs':{}}
            for spec in specs:
                signal_type=spec.split('_')[0];signal=signals[signal_type]
                result=[]
                for c in THRESHOLDS:
                    if min(np.sum(t<=c),np.sum(t>c))<max(50,.04*len(t)):continue
                    if spec.endswith('share_fit'):
                        key='all' if signal_type=='postcode' else 'weighted'
                        X=share_design(t,censor,c,s[f'share_{key}_{int(c)}'].to_numpy())
                    else:X=base.step_design(t,censor,c)
                    beta,scores,corr=model_stats(X,pseudo,blocks)
                    sigbeta,sigscores,_=model_stats(X,signal[:,None],blocks)
                    se=np.sqrt(corr*np.sum(scores**2,axis=0))
                    result.append({'c':c,'beta':beta,'scores':scores,'corr':corr,'sigbeta':float(sigbeta[0]),'sigscores':sigscores,'se':se})
                    if c==15:
                        analytic.append({'study':city,'relation':r,'spec':spec,'conditional_signal_coefficient':float(sigbeta[0]),'signal_sd':float(np.std(signal,ddof=1)),'residual_signal_sd':float(np.std(signal-X[:,[0,1,3,4]]@np.linalg.lstsq(X[:,[0,1,3,4]],signal,rcond=None)[0],ddof=1))})
                        if spec.endswith('share_fit'):assert abs(float(sigbeta[0])-1)<1e-7
                nullmax=np.max(np.abs(np.stack([z['beta'][:NCAL]/z['se'][:NCAL] for z in result],axis=1)),axis=1)
                panel['specs'][spec]=(result,nullmax)
            panels.append(panel)
            print('Injection prepared',city,r,flush=True)
    rows=[];by_panel=[];nullrows=[]
    for spec in specs:
        for mag in (0.,.5,1.,1.5,2.):
            for direction in ((1,) if mag==0 else (-1,1)):
                delta=mag*direction;pvalues=[];step15=[];peaks=[]
                for p in panels:
                    fits,nullmax=p['specs'][spec];statistics=[];estimates=[]
                    for z in fits:
                        beta=z['beta'][NCAL:]+delta*z['sigbeta']
                        scores=z['scores'][:,NCAL:]+delta*z['sigscores']
                        se=np.sqrt(z['corr']*np.sum(scores**2,axis=0))
                        statistics.append(beta/se);estimates.append(beta)
                    statistics=np.column_stack(statistics);estimates=np.column_stack(estimates)
                    pvalues.append(old.empirical_tail_p(nullmax,np.max(np.abs(statistics),axis=1)))
                    ix=[z['c'] for z in fits].index(15.)
                    step15.append(estimates[:,ix]);peaks.append(np.argmax(np.abs(statistics),axis=1)==ix)
                q=base.bh_fdr_rows(np.column_stack(pvalues));selected=q<=.05
                est=np.column_stack(step15)
                row={'spec':spec,'absolute_step':mag,'direction':direction,'n_calibration':NCAL,'n_evaluation':NSIM,'mean_panel_detection':float(selected.mean()),'probability_any_panel':float(selected.any(axis=1).mean()),'mean_estimated_step15':float(est.mean()),'mean_ratio':float(est.mean()/delta) if delta else np.nan}
                rows.append(row)
                if mag==0:nullrows.append(row)
                for j,p in enumerate(panels):by_panel.append({**row,'study':p['study'],'relation':p['relation'],'detection':float(selected[:,j].mean()),'estimated_step15':float(est[:,j].mean()),'ratio':float(est[:,j].mean()/delta) if delta else np.nan,'rmse_step15':float(np.sqrt(np.mean((est[:,j]-delta)**2)))})
    pd.DataFrame(rows).to_csv(OUT/'remedy_injection_summary.csv',index=False)
    pd.DataFrame(by_panel).to_csv(OUT/'remedy_injection_by_panel.csv',index=False)
    pd.DataFrame(analytic).to_csv(OUT/'analytic_signal_projection.csv',index=False)
    pd.DataFrame(nullrows).to_csv(OUT/'independent_null_calibration.csv',index=False)
    print(pd.DataFrame(rows).query('absolute_step==1.5').groupby('spec')[['mean_panel_detection','mean_ratio']].mean().to_string())

def main():
    studies={city:base.load_study(ROOT,city,25,SEED+i) for i,city in enumerate(base.STUDIES)}
    for city,b in studies.items():np.savez_compressed(OUT/f'{city}_analysis_inputs.npz',**{'codes':b['codes'],'blocks':b['blocks'],'observed':b['observed'],'context_residual':b['residuals']['pooled_context_hgb'],'hfge_residual':b['residuals']['hfge']})
    summaries=pd.read_csv(OUT/'postcode_summaries.csv')
    flexible_scan(studies,summaries)
    injection(studies,summaries)
    (OUT/'inference_settings.json').write_text(json.dumps({'spatial_blocks':25,'block_seed':SEED,'bootstrap_replicates':NBOOT,'calibration_replicates':NCAL,'evaluation_replicates':NSIM,'candidate_minutes':THRESHOLDS,'splines':{'df4':[0,5,15,25,40],'df6':[0,5,10,15,20,30,40]},'null_calibration':'Separate calibration and evaluation pseudo-outcomes; spatial wild signs; BH across four city-relation panels within each specification.'},indent=2))

if __name__=='__main__':main()
