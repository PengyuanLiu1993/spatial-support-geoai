"""Reroute postcode and ONS origins on the frozen networks; assess weighted coverage."""
from pathlib import Path
import sys, json
import numpy as np
import pandas as pd
import geopandas as gpd
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'analysis'))
import run_origin_sensitivity_audit as old

OUT=ROOT/'results/revision_20260925'
OUT.mkdir(parents=True,exist_ok=True)
RAW=ROOT/'data/raw/revision_20260925'
CITIES=('greater_london','greater_manchester')
RELATIONS=('nhs_gp','osm_pharmacy')
THRESHOLDS=(5,10,15,20)

def weighted_quantile(values,weights,q):
    order=np.argsort(values)
    v,w=np.asarray(values)[order],np.asarray(weights)[order]
    return float(v[min(np.searchsorted(np.cumsum(w),q*w.sum(),side='left'),len(v)-1)])

def route(city):
    cache=OUT/f'{city}_routed_origins.parquet'
    case=ROOT/f'data/processed/case_studies/{city}'
    lsoas=gpd.read_parquet(case/'lsoas.parquet').to_crs(27700)
    if cache.exists():return pd.read_parquet(cache),lsoas
    codes=lsoas.lsoa21cd.astype(str).to_numpy()
    codepoints=old.load_codepoints(ROOT/'data/raw/code_point_open_2026_08/extracted/Data/CSV',tuple(lsoas.total_bounds))
    loc=old.PolygonGridLocator(lsoas.geometry)
    print(city,'assigning postcode origins',flush=True)
    assigned=[loc.locate(float(x),float(y)) for x,y in codepoints[['x','y']].itertuples(index=False)]
    assigned=np.array([-1 if i is None else i for i in assigned])
    pc=codepoints.loc[assigned>=0].copy()
    pc['lsoa21cd']=codes[assigned[assigned>=0]]
    pc['origin_type']='postcode_centres'
    rep=lsoas.geometry.representative_point()
    rep=pd.DataFrame({'lsoa21cd':codes,'x':rep.x,'y':rep.y,'origin_type':'representative_point'})
    pwc=pd.read_csv(RAW/f'{city}_ons_population_weighted_centroids.csv')
    pwc['origin_type']='ons_population_weighted_centroid'
    origins=pd.concat([pc,rep,pwc],ignore_index=True)
    graph,xy=old.load_road_graph(case/'walk_network.npz')
    gp=old.load_gp_facilities(ROOT/'data/raw/nhs_ods_2026_08_26/gp_practices.csv',codepoints)
    ph=gpd.read_parquet(case/'osm_facilities.parquet').to_crs(27700)
    ph=ph[ph.category.eq('pharmacy')]
    facilities={'nhs_gp':gp,'osm_pharmacy':ph}
    distances={}
    for r,frame in facilities.items():
        print(city,'routing',r,flush=True)
        distances[r],_=old.distance_to_nearest_facility(graph,xy,frame)
    origins=old.attach_times(origins,xy,cKDTree(xy),distances)
    origins.to_parquet(cache,index=False)
    return origins,lsoas

def main():
    p=pd.read_csv(RAW/'census2021_postcode_residents_p001.csv')
    p['postcode']=p.Postcode.str.upper().str.replace(' ','',regex=False)
    pop=p.groupby('postcode')['Count'].sum()
    summaries=[]; comparisons=[]; metadata=[]; random_rows=[]; original_audit=[]
    for city_idx,city in enumerate(CITIES):
        routed,lsoas=route(city)
        codes=lsoas.lsoa21cd.astype(str).to_numpy()
        pc=routed[routed.origin_type.eq('postcode_centres')].copy()
        pc['census2021_residents']=pc.postcode.map(pop)
        pc.to_parquet(OUT/f'{city}_postcode_population_routes.parquet',index=False)
        metadata.append({'study':city,'n_lsoas':len(codes),'n_current_postcodes':len(pc),'n_matched_postcodes':int(pc.census2021_residents.notna().sum()),'matched_postcode_percent':100*pc.census2021_residents.notna().mean(),'matched_residents':int(pc.census2021_residents.sum()),'lsoas_with_population':int(pc[pc.census2021_residents.gt(0)].lsoa21cd.nunique()),'area_km2_p10':float(lsoas.area.quantile(.1)/1e6),'area_km2_median':float(lsoas.area.median()/1e6),'area_km2_p90':float(lsoas.area.quantile(.9)/1e6)})
        for r in RELATIONS:
            a=old.aggregate_origins(routed[routed.origin_type.isin(['postcode_centres','representative_point'])],codes,r)
            a.insert(0,'study',city);original_audit.append(a)
        city_rows=[]
        for code,group in pc.groupby('lsoa21cd',sort=True):
            weights=group.census2021_residents.fillna(0).to_numpy(float)
            for r in RELATIONS:
                t=group[f'{r}_minutes'].to_numpy(float)
                capped=np.minimum(t,40.)
                row={'study':city,'lsoa21cd':code,'relation':r,'n_postcodes':len(group),'n_matched_postcodes':int((weights>0).sum()),'population':float(weights.sum()),'median_unweighted':float(np.median(capped)),'p90_unweighted':float(np.quantile(capped,.9))}
                for q in (.1,.5,.9):
                    row[f'weighted_q{int(q*100)}']=weighted_quantile(capped,weights,q) if weights.sum()>0 else np.nan
                for c in THRESHOLDS:
                    row[f'share_all_{c}']=float(np.mean(t<=c))
                    row[f'share_matched_equal_{c}']=float(np.mean(t[weights>0]<=c)) if (weights>0).any() else np.nan
                    row[f'share_weighted_{c}']=float(np.average(t<=c,weights=weights)) if weights.sum()>0 else np.nan
                city_rows.append(row)
        summary=pd.DataFrame(city_rows);summaries.append(summary)
        for r in RELATIONS:
            s=summary[summary.relation.eq(r)].set_index('lsoa21cd').reindex(codes)
            for origin in ('representative_point','ons_population_weighted_centroid'):
                t=routed[routed.origin_type.eq(origin)].set_index('lsoa21cd').reindex(codes)[f'{r}_minutes'].to_numpy(float)
                for benchmark in ('all','weighted','matched_equal'):
                    for c in THRESHOLDS:
                        shares=s[f'share_{benchmark}_{c}'].to_numpy(float)
                        valid=np.isfinite(shares)
                        target=shares[valid]>=.5; point=t[valid]<=c
                        into=int(np.sum(~point&target));outof=int(np.sum(point&~target))
                        comparisons.append({'study':city,'relation':r,'origin':origin,'benchmark':benchmark,'threshold':c,'n_lsoas':int(valid.sum()),'point_uncovered_to_majority_covered':into,'point_covered_to_majority_uncovered':outof,'flip_count':into+outof,'flip_percent':100*(into+outof)/valid.sum(),'population_coverage_share':float(np.average(shares[valid],weights=s.population.to_numpy()[valid])) if benchmark=='weighted' else np.nan,'median_absolute_share_error':float(np.median(np.abs(point-shares[valid])))})
        # One postcode is drawn per LSOA per replicate; the same draw is used for both relations and all bands.
        rng=np.random.default_rng(20260925+city_idx)
        counts={(draw,b,r,c):np.zeros((1000,2),int) for draw in ('uniform_postcode','population_proportional_postcode') for b in ('all','weighted') for r in RELATIONS for c in THRESHOLDS}
        nvalid={k:0 for k in counts}
        indexed=summary.set_index(['lsoa21cd','relation'])
        for code,g in pc.groupby('lsoa21cd',sort=True):
            u=rng.random(1000)
            w=g.census2021_residents.fillna(0).to_numpy(float)
            indices={'uniform_postcode':np.minimum((u*len(g)).astype(int),len(g)-1)}
            if w.sum()>0:indices['population_proportional_postcode']=np.minimum(np.searchsorted(np.cumsum(w)/w.sum(),u,side='right'),len(w)-1)
            for draw,ix in indices.items():
                for r in RELATIONS:
                    t=g[f'{r}_minutes'].to_numpy(float)[ix]
                    for b in ('all','weighted'):
                        for c in THRESHOLDS:
                            share=indexed.loc[(code,r),f'share_{b}_{c}']
                            if not np.isfinite(share):continue
                            key=(draw,b,r,c);nvalid[key]+=1
                            target=share>=.5;point=t<=c
                            counts[key][:,0]+=(~point)&target
                            counts[key][:,1]+=point&(~target)
        for (draw,b,r,c),v in counts.items():
            n=nvalid[(draw,b,r,c)];rates=100*v.sum(axis=1)/n
            random_rows.append({'study':city,'relation':r,'draw':draw,'benchmark':b,'threshold':c,'n_lsoas':n,'replicates':1000,'mean_flip_percent':float(rates.mean()),'p025_flip_percent':float(np.quantile(rates,.025)),'p975_flip_percent':float(np.quantile(rates,.975)),'mean_into_count':float(v[:,0].mean()),'mean_out_count':float(v[:,1].mean())})
        print(city,'audit finished',flush=True)
    pd.concat(summaries).to_csv(OUT/'postcode_summaries.csv',index=False)
    pd.DataFrame(comparisons).to_csv(OUT/'origin_comparator_counts.csv',index=False)
    pd.DataFrame(random_rows).to_csv(OUT/'random_postcode_benchmark.csv',index=False)
    pd.DataFrame(metadata).to_csv(OUT/'population_matching_and_area.csv',index=False)
    pd.concat(original_audit).to_csv(OUT/'reconstructed_postcode_origin_audit.csv',index=False)
    new=old.summarise_comparison(pd.concat(original_audit))
    prior=pd.read_csv(ROOT/'reference_results/paper/origin_sensitivity/origin_sensitivity_summary.csv')
    joined=new.merge(prior,on=['study','relation','origin_type','threshold_minutes'],suffixes=('_new','_old'))
    assert len(joined)==16
    assert np.array_equal(joined.classification_flip_count_new,joined.classification_flip_count_old)
    print('Reproduced all 16 original postcode classification counts exactly.',flush=True)
    pd.DataFrame(comparisons).query("threshold==15 and benchmark=='weighted'").to_csv(OUT/'primary_15min_comparators.csv',index=False)
    print(pd.DataFrame(comparisons).query("threshold==15 and benchmark=='weighted'")[['study','relation','origin','flip_count','flip_percent']].to_string(index=False))

if __name__=='__main__':main()
