"""Run the graph component with the manuscript's frozen hyperparameters."""
from pathlib import Path
import argparse
import json
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device',default='cuda')
    parser.add_argument('--dry-run',action='store_true')
    args=parser.parse_args()
    config=json.loads((ROOT/'configs/paper_graph.json').read_text(encoding='utf-8'))
    command=[sys.executable,str(ROOT/'scripts/run_heterogeneous_facility_gnn.py')]
    for key,value in config.items():
        flag='--'+key.replace('_','-')
        if isinstance(value,bool):command.append(flag if value else '--no-'+key.replace('_','-'))
        else:command.extend([flag,','.join(map(str,value)) if isinstance(value,list) else str(value)])
    command.extend(['--device',args.device])
    if args.dry_run:
        print(json.dumps(command[2:],indent=2))
    else:subprocess.run(command,cwd=ROOT,check=True)

if __name__=='__main__':main()
