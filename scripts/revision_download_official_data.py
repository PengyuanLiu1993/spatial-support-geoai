"""Download public ONS centroid and postcode population inputs for the revision."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import requests
import pandas as pd
import geopandas as gpd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'data/raw/revision_20260925'
OUT.mkdir(parents=True, exist_ok=True)
SERVICE = 'https://services1.arcgis.com/ESMARspQHYMw9BZ9/arcgis/rest/services/LSOA_PopCentroids_EW_2021_V4/FeatureServer/0'
POP = 'https://www.nomisweb.co.uk/output/census/2021/pcd_p001.csv'
manifest = []
for city in ('greater_london','greater_manchester'):
    path = OUT / f'{city}_ons_population_weighted_centroids.csv'
    if not path.exists():
        codes = gpd.read_parquet(ROOT / f'data/processed/case_studies/{city}/lsoas.parquet')['lsoa21cd'].tolist()
        rows = []
        for start in range(0,len(codes),80):
            subset = codes[start:start+80]
            where = 'LSOA21CD IN (' + ','.join("'"+c+"'" for c in subset) + ')'
            response = requests.get(SERVICE+'/query', params={'f':'json','where':where,'outFields':'LSOA21CD','outSR':27700,'returnGeometry':'true'}, timeout=60)
            response.raise_for_status()
            payload = response.json()
            assert 'error' not in payload, payload
            assert not payload.get('exceededTransferLimit',False)
            for f in payload['features']:
                rows.append({'lsoa21cd':f['attributes']['LSOA21CD'],'x':f['geometry']['x'],'y':f['geometry']['y']})
            print(city, len(rows), flush=True)
        frame = pd.DataFrame(rows)
        assert len(frame)==len(codes) and frame.lsoa21cd.is_unique
        frame.to_csv(path,index=False)
    manifest.append({'file':path.name,'source':SERVICE,'sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
path = OUT / 'census2021_postcode_residents_p001.csv'
if not path.exists():
    response = requests.get(POP, timeout=120)
    response.raise_for_status()
    assert len(response.content)>1_000_000
    path.write_bytes(response.content)
manifest.append({'file':path.name,'source':POP,'sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
(OUT/'sources.json').write_text(json.dumps({'retrieved_utc':datetime.now(timezone.utc).isoformat(),'files':manifest},indent=2),encoding='utf-8')
print('Downloads complete',flush=True)
print(pd.read_csv(path,nrows=8).to_string(index=False))
