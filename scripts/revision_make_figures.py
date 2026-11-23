from pathlib import Path
import sys, os, json
os.environ.setdefault('MPLCONFIGDIR',str(Path.cwd()/'work/.matplotlib'))
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Polygon as MplPolygon
from matplotlib.collections import LineCollection
import osmium
from shapely import from_wkb
from shapely.ops import transform
from pyproj import Transformer

ROOT=Path(__file__).resolve().parents[1]
P=ROOT/'results/revision_20260925';OUT=P/'figures';OUT.mkdir(exist_ok=True)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.titlesize':11,'axes.labelsize':10,'legend.fontsize':9,'svg.fonttype':'none','pdf.fonttype':42})
CITIES=('greater_london','greater_manchester');LABELS=('London','Manchester')
BLUE='#2166ac';ORANGE='#d6604d';GREEN='#1b7837';GRAY='#68727b'

def save(fig,name):
    for ext in ('png','pdf','svg'):fig.savefig(OUT/f'{name}.{ext}',dpi=300,bbox_inches='tight',facecolor='white')
    plt.close(fig)

def box(ax,x,y,w,h,text,fill='#f0f4f7',fontsize=10):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.02,rounding_size=0.02',facecolor=fill,edgecolor='#bdc8cf',lw=.8))
    ax.text(x+w/2,y+h/2,text,ha='center',va='center',fontsize=fontsize,linespacing=1.4)

def arrow(ax,a,b,color=GRAY):ax.add_patch(FancyArrowPatch(a,b,arrowstyle='-|>',mutation_scale=12,color=color,lw=1.1))

def framework():
    # The five-panel framework uses the supplied vector source. Build it with the
    # bundled document runtime; do not regenerate the withdrawn replacement.
    required=[OUT/f'figure2_hfge_revised.{ext}' for ext in ('png','pdf','svg')]
    if not all(p.exists() for p in required):
        raise RuntimeError('Run scripts/revision_relayout_original_framework.py with the figure dependencies installed first.')


class LocalBuildings(osmium.SimpleHandler):
    def __init__(self,bounds):
        super().__init__();self.bounds=bounds;self.factory=osmium.geom.WKBFactory();self.proj=Transformer.from_crs(4326,27700,always_xy=True);self.geoms=[];self.tags=[]
    def area(self,area):
        tag=area.tags.get('building')
        if not tag:return
        try:
            geom=from_wkb(bytes.fromhex(self.factory.create_multipolygon(area)))
            lon,lat=geom.representative_point().coords[0]
            x,y=self.proj.transform(lon,lat)
            if not (self.bounds[0]<=x<=self.bounds[2] and self.bounds[1]<=y<=self.bounds[3]):return
            self.geoms.append(transform(self.proj.transform,geom));self.tags.append(str(tag))
        except (RuntimeError,ValueError):return

def geography():
    fig=plt.figure(figsize=(7.2,8.6))
    grid=fig.add_gridspec(3,2,height_ratios=[1.15,.72,.32],hspace=.54,wspace=.34)
    metadata=[]
    for ci,(city,label) in enumerate(zip(CITIES,LABELS)):
        lsoas=gpd.read_parquet(ROOT/f'data/processed/case_studies/{city}/lsoas.parquet').set_index('lsoa21cd')
        s=pd.read_csv(P/'postcode_summaries.csv');s=s[s.study.eq(city)&s.relation.eq('nhs_gp')].set_index('lsoa21cd')
        area=lsoas.area
        eligible=s[s.share_weighted_15.between(.25,.75)&s.n_matched_postcodes.ge(10)].index
        code=(area.loc[eligible]-area.median()).abs().idxmin()
        poly=lsoas.loc[code].geometry;ctr=poly.representative_point();cx,cy=ctr.x,ctr.y
        bounds=(cx-800,cy-800,cx+800,cy+800)
        cache=P/f'{city}_illustration_buildings.parquet'
        if cache.exists():buildings=gpd.read_parquet(cache)
        else:
            filename='greater_london_2026_08_20/greater-london-260820.osm.pbf' if ci==0 else 'greater_manchester_2026_08_26/greater-manchester-260826.osm.pbf'
            handler=LocalBuildings(bounds);print('Extracting map buildings',city,code,flush=True)
            handler.apply_file(str(ROOT/'data/raw/osm'/filename),locations=True,idx='flex_mem')
            buildings=gpd.GeoDataFrame({'building':handler.tags},geometry=handler.geoms,crs=27700);buildings.to_parquet(cache)
        ax=fig.add_subplot(grid[0,ci]);ax.set_title(f'{chr(97+ci)}  {label} example',loc='left',fontweight='bold')
        near=lsoas.cx[bounds[0]:bounds[2],bounds[1]:bounds[3]]
        near.plot(ax=ax,facecolor='#fafafa',edgecolor='#b8bdc1',linewidth=.65)
        buildings.plot(ax=ax,facecolor='#cbd1d5',edgecolor='none')
        gpd.GeoSeries([poly],crs=27700).plot(ax=ax,facecolor='none',edgecolor='#202a31',linewidth=1.5)
        pc=pd.read_parquet(P/f'{city}_postcode_population_routes.parquet')
        allpc=pc[pc.x.between(bounds[0],bounds[2])&pc.y.between(bounds[1],bounds[3])]
        ax.scatter(allpc.x,allpc.y,c='#9fb7cc',s=6,alpha=.6,zorder=4)
        chosen=pc[pc.lsoa21cd.eq(code)]
        inside=chosen.nhs_gp_minutes.le(15)
        ax.scatter(chosen.x,chosen.y,c=np.where(inside,BLUE,ORANGE),s=16,edgecolors='white',linewidths=.2,zorder=5)
        pwc=pd.read_csv(ROOT/f'data/raw/revision_20260925/{city}_ons_population_weighted_centroids.csv').set_index('lsoa21cd').loc[code]
        ax.scatter([cx],[cy],marker='x',s=85,color='black',lw=1.8,zorder=6)
        ax.scatter([pwc.x],[pwc.y],marker='*',s=150,color=GREEN,edgecolor='white',lw=.6,zorder=7)
        ax.plot([bounds[0]+80,bounds[0]+580],[bounds[1]+85]*2,color='black',lw=2)
        ax.text(bounds[0]+330,bounds[1]+120,'500 m',ha='center',fontsize=8)
        ax.set(xlim=(bounds[0],bounds[2]),ylim=(bounds[1],bounds[3]),aspect='equal');ax.axis('off')
        ax.text(.02,-.05,f'{code}  |  {poly.area/1e6:.2f} km²\n{len(chosen)} postcode points',transform=ax.transAxes,fontsize=9,va='top')
        ax=fig.add_subplot(grid[1,ci])
        ax.set_title(f'{chr(99+ci)}  GP walking-time distribution',loc='left',fontweight='bold',fontsize=10)
        matched=chosen.dropna(subset=['census2021_residents']).sort_values('nhs_gp_minutes')
        t=np.minimum(matched.nhs_gp_minutes.to_numpy(),40);w=matched.census2021_residents.to_numpy()
        ax.step(t,100*np.cumsum(w)/w.sum(),where='post',color=BLUE,lw=2,label='Population-weighted share')
        ax.axvline(15,color=ORANGE,ls='--',lw=1.3)
        share=100*s.loc[code,'share_weighted_15'];ax.scatter([15],[share],c=ORANGE,zorder=5)
        ax.text(15.5,max(share-12,7),f'{share:.0f}% within 15 min',fontsize=8,color=ORANGE)
        ax.set(xlim=(0,max(25,min(40,t.max()+2))),ylim=(0,102),xlabel='Walking time to nearest GP (minutes)',ylabel='Represented population (%)')
        ax.spines[['top','right']].set_visible(False);ax.grid(axis='y',alpha=.2)
        metadata.append({'study':city,'lsoa21cd':code,'area_km2':poly.area/1e6,'postcodes':len(chosen),'weighted_share_gp15':share/100,'selection':'Among LSOAs with 25–75% weighted GP coverage and at least ten matched postcodes, closest to city median area.'})
    ax=fig.add_subplot(grid[2,:]);ax.axis('off')
    ax.scatter([.005,.275],[.83,.83],c=[BLUE,ORANGE],s=28,transform=ax.transAxes)
    ax.text(.035,.83,'Postcode ≤15 min',transform=ax.transAxes,va='center',fontsize=9)
    ax.text(.295,.83,'Postcode >15 min',transform=ax.transAxes,va='center',fontsize=9)
    ax.scatter([.56],[.83],marker='x',color='black',s=55,transform=ax.transAxes)
    ax.text(.58,.83,'Interior point',transform=ax.transAxes,va='center',fontsize=9)
    ax.scatter([.76],[.83],marker='*',color=GREEN,s=85,transform=ax.transAxes)
    ax.text(.78,.83,'ONS centre',transform=ax.transAxes,va='center',fontsize=9)
    ax.text(.01,.35,'Grey footprints = mapped buildings; dark outline = selected LSOA.\nPostcodes group delivery addresses. A postcode point is not a building or a resident.',transform=ax.transAxes,fontsize=9,va='center',linespacing=1.5)
    save(fig,'figure3_geographic_scale')
    (P/'map_examples.json').write_text(json.dumps(metadata,indent=2))

def audit_figure():
    counts=pd.read_csv(P/'origin_comparator_counts.csv');random=pd.read_csv(P/'random_postcode_benchmark.csv')
    fig,axes=plt.subplots(2,2,figsize=(7.1,5.4),sharex=True,sharey=True)
    for ci,(city,label) in enumerate(zip(CITIES,LABELS)):
        for ri,(r,rl) in enumerate([('nhs_gp','GP'),('osm_pharmacy','Pharmacy')]):
            ax=axes[ci,ri]
            for origin,color,lab,marker in [('representative_point',ORANGE,'Interior point','s'),('ons_population_weighted_centroid',BLUE,'ONS population centre','o')]:
                d=counts[counts.study.eq(city)&counts.relation.eq(r)&counts.benchmark.eq('weighted')&counts.origin.eq(origin)].sort_values('threshold')
                ax.plot(d.threshold,d.flip_percent,label=lab,c=color,marker=marker,lw=1.8)
            d=random[random.study.eq(city)&random.relation.eq(r)&random.benchmark.eq('weighted')&random.draw.eq('uniform_postcode')].sort_values('threshold')
            ax.plot(d.threshold,d.mean_flip_percent,label='Random postcode',c=GRAY,ls='--',marker='^',lw=1.4)
            ax.fill_between(d.threshold,d.p025_flip_percent,d.p975_flip_percent,color=GRAY,alpha=.13)
            ax.set_title(label+' '+rl,loc='left');ax.set_xticks([5,10,15,20]);ax.spines[['top','right']].set_visible(False);ax.grid(axis='y',alpha=.2)
            if ri==0:ax.set_ylabel('LSOAs with different\nclassification (%)')
            if ci==1:ax.set_xlabel('Walking-time band (minutes)')
    handles,labels=axes[0,0].get_legend_handles_labels();fig.legend(handles,labels,loc='lower center',ncol=3,fontsize=8.5,frameon=False)
    fig.tight_layout(rect=(0,.065,1,1));save(fig,'figure4_origin_comparison')

def injection_figure():
    d=pd.read_csv(P/'remedy_injection_summary.csv');d=d[d.absolute_step.gt(0)].groupby(['spec','absolute_step']).mean(numeric_only=True).reset_index()
    fig,axes=plt.subplots(2,1,figsize=(7.1,6.4),gridspec_kw={'height_ratios':[1.15,1]})
    specs=[('point_signal_point_fit','Point signal / point fit','#999999','--','.'),('postcode_signal_point_fit','Postcode signal / point fit',ORANGE,'-','s'),('postcode_signal_share_fit','Postcode signal / share fit',BLUE,'-','o'),('population_signal_share_fit','Population signal / weighted-share fit',GREEN,'-','^')]
    ax=axes[0]
    for spec,label,color,ls,m in specs:
        v=d[d.spec.eq(spec)];ax.plot(v.absolute_step,100*v.mean_panel_detection,label=label,color=color,ls=ls,marker=m,lw=1.8)
    ax.set(xlabel='Injected origin-level change (AHAH points)',ylabel='Mean panel detection (%)',ylim=(0,103),xticks=[.5,1,1.5,2]);ax.set_title('a  Recovery improves when coverage shares enter the model',loc='left',fontsize=11)
    ax.legend(frameon=False,fontsize=8.5,loc='upper left');ax.grid(alpha=.2);ax.spines[['top','right']].set_visible(False)
    a=pd.read_csv(P/'analytic_signal_projection.csv');panels=[(c,r) for c in CITIES for r in ('nhs_gp','osm_pharmacy')]
    ax=axes[1];x=np.arange(4)
    for k,(spec,label,color) in enumerate([('postcode_signal_point_fit','Postcode signal / point fit',ORANGE),('population_signal_point_fit','Population signal / point fit','#c19748'),('population_signal_share_fit','Population signal / weighted-share fit',GREEN)]):
        vals=[a[a.study.eq(c)&a.relation.eq(r)&a.spec.eq(spec)].conditional_signal_coefficient.iloc[0] for c,r in panels]
        ax.bar(x+(k-1)*.24,vals,.23,label=label,color=color)
    ax.set(xticks=x,xticklabels=['London\nGP','London\nPharmacy','Manchester\nGP','Manchester\nPharmacy'],ylabel='Expected coefficient / injected change',ylim=(0,1.13));ax.axhline(1,color='gray',ls=':',lw=1)
    ax.set_title('b  A matching coverage-share predictor removes coefficient attenuation',loc='left',fontsize=10.4)
    ax.legend(frameon=False,fontsize=8,loc='upper center',bbox_to_anchor=(.5,-.20),ncol=2);ax.spines[['top','right']].set_visible(False)
    fig.tight_layout();save(fig,'figure5_signal_recovery')

if __name__=='__main__':
    framework();audit_figure();injection_figure();geography()
