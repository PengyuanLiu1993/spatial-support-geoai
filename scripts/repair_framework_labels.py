from pathlib import Path
from lxml import etree as E
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
import json, re, os

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/figure3_original_repaired';OUT.mkdir(exist_ok=True)
SRC=ROOT/'assets/framework/two_row.svg'
S='http://www.w3.org/2000/svg';N={'s':S}
r=E.parse(str(SRC));root=r.getroot()
pdfmetrics.registerFont(TTFont('Arial',os.environ.get('HFGE_ARIAL', 'arial.ttf')))
pdfmetrics.registerFont(TTFont('Arial-Bold',os.environ.get('HFGE_ARIAL_BOLD', 'arialbd.ttf')))
original_paths=[E.tostring(p) for p in root.xpath('//s:path',namespaces=N)]
for t in root.xpath('//s:text',namespaces=N):t.getparent().remove(t)
def transform_group(gid,transform):
    root.xpath(f'//s:g[@id="{gid}"]',namespaces=N)[0].set('transform',transform)
transform_group('patch_8','translate(0 370.419285) scale(1 1.4) translate(0 -370.419285)')
transform_group('patch_36','translate(0 399.564597) scale(1 1.7) translate(0 -399.564597)')
transform_group('patch_341','translate(1019.4804 0) scale(1.25 1) translate(-1019.4804 0)')
transform_group('patch_385','translate(0 415.904147) scale(1 0.4) translate(0 -406.904147)')
layer=E.SubElement(root,'{'+S+'}g',id='repaired_labels')
P={'a':(12,44,181.44),'b':(217.44,44,325.08),'c':(566.52,44,294.84),'d':(140.424,565,325.08),'e':(489.504,565,243.432)}
labels=[]
COL={'ink':'#1c2532','muted':'#526170','area':'#62788c','model':'#27755f','gp':'#377e9f','h':'#c45a50','g':'#3e8965','f':'#d88a3d','tab':'#6e63a8'}
def txt(panel,x,y,value,size=14.5,bold=False,anchor='middle',colour='ink',step=16.3):
    ox,oy,w=P.get(panel,(0,0,873.36));font='Arial-Bold' if bold else 'Arial'
    for n,line in enumerate(value.split('\n')):
        width=pdfmetrics.stringWidth(line,font,size);xx=ox+x;yy=oy+y+n*step
        left=xx if anchor=='start' else xx-width if anchor=='end' else xx-width/2
        assert left>=ox-1 and left+width<=ox+w+1,(panel,line,left-ox,width,w)
        t=E.SubElement(layer,'{'+S+'}text',x=str(xx),y=str(yy),style=f'font-family: Arial; font-size: {size}px; font-weight: {700 if bold else 400}; text-anchor: {anchor}; fill: {COL.get(colour,colour)}')
        t.text=line
        labels.append({'panel':panel,'text':line,'size':size,'bbox':[left,yy-size*.74,left+width,yy+size*.2]})
def circ(panel,x,y,label,colour,radius=10):
    ox,oy,_=P[panel]
    E.SubElement(layer,'{'+S+'}circle',cx=str(ox+x),cy=str(oy+y),r=str(radius),fill=COL[colour])
    txt(panel,x,y+5,label,bold=True,colour='white')

txt(None,436.68,20,'Heterogeneous Facility Graph Ensemble (HFGE)',size=19,bold=True)
for p,title in [('a','Spatial inputs'),('b','Spatial heterogeneous graph'),('c','Typed relation encoder'),('d','Heterogeneous facility GNN'),('e','Two-stage HFGE prediction')]:
    txt(p,7,25,f'({p})  {title}',size=15.5,bold=True,anchor='start')

# All original map, network, bar, edge, box and arrow paths are retained.
# Labels use the existing two-row geometry, with local wrapping and spacing.
txt('a',90,64,'Case-study geography',bold=True)
txt('a',48,84,'London',bold=True)
txt('a',132,84,'Manchester',bold=True)
txt('a',17,201,'Node attributes',bold=True,anchor='start')
txt('a',37,278,'103')
txt('a',92,242,'GP');txt('a',137,242,'Hospital')
txt('a',95,282,'Green');txt('a',146,282,'Food')
txt('a',17,334,'Spatial evaluation',bold=True,anchor='start')
txt('a',127,365,'Fit on\ntraining\nfolds; test\nheld-out\nfold')
txt('a',77,473,'5 contiguous folds')

txt('b',162.5,66,'Facility access and area adjacency',bold=True)
txt('b',54,408,'Facility edge: type,',anchor='start')
txt('b',54,424,'walking time and mass',anchor='start')
txt('b',54,447,'Queen edge: shared',anchor='start')
txt('b',54,463,'boundary or corner',anchor='start')
for x,y,s,c in [(220.72,205.0,'GP','gp'),(194.43,87.7,'H','h'),(172.59,279.6,'G','g'),(152.84,263.2,'F','f')]:circ('b',x,y,s,c,12 if s=='GP' else 10)
ox,oy,_=P['b']
E.SubElement(layer,'{'+S+'}polyline',points=' '.join(f'{ox+x},{oy+y}' for x,y in [(165.6,197.3),(244,309),(250,309)]),fill='none',stroke=COL['ink'],**{'stroke-width':'0.9'})
txt('b',254,315,'Target\narea',anchor='start',bold=True)

txt('c',38,61,'Edge\nstate',colour='muted')
txt('c',88,68,'MLP',colour='muted');txt('c',159,68,'Decay',colour='muted')
txt('c',203,68,'Pool',colour='muted');txt('c',252,68,'Vector',colour='muted')
for j,col in enumerate(['gp','h','g','f']):
    y=123.7+j*71.6688
    circ('c',14.742,y,['GP','H','G','F'][j],col,12 if j==0 else 10)
    txt('c',125,y+37,'Apply',colour='ink')
    txt('c',203.44,y+37,'Pool',colour=col)
txt('c',147,392,'Time, mass and nonlinear terms',colour='muted')
txt('c',185,437,'Relation-specific\nfacility vectors',bold=True,colour='model')

txt('d',55,69,'Area attributes',bold=True,colour='area')
txt('d',233,69,'Facility relations',bold=True,colour='model')
txt('d',68,137,'Encoder',colour='area')
txt('d',119,137,'State',colour='area')
txt('d',246,137,'Encoder',colour='model')
txt('d',294,137,'State',colour='model')
txt('d',163,137,'Join',colour='model',bold=True)
txt('d',78,189,'Joint-fusion\nMLP',colour='model')
txt('d',135,241,'Joint state',colour='model',anchor='end')
txt('d',172,273,'Queen-neighbour message passing',colour='model',bold=True)
txt('d',39,340,'Queen\ngraph',colour='area')
txt('d',119,340,'GraphConv\nlayer 1',colour='model')
txt('d',207,340,'GraphConv\nlayer 2',colour='model')
txt('d',286,340,'Graph\nstate',colour='model')
txt('d',36,377,'Joint',colour='model');txt('d',86,377,'Graph',colour='model')
txt('d',176,377,'Context',colour='model');txt('d',232,377,'Skip',colour='area')
txt('d',140,396,'Gate',colour='model',bold=True)
txt('d',240,416,'Join',colour='model',bold=True)
txt('d',272,436,'Readout',colour='model')
txt('d',296,451,'GNN\nresidual',colour='model',bold=True)
txt('d',132,454,'State bars show\ngated skip connections',colour='muted')

txt('e',16,66,'1  Residual graph hybrid',bold=True,anchor='start')
txt('e',48,156,'Context-HGB\nprediction',colour='tab')
txt('e',190,147,'Weighted\nGNN residual',colour='model')
txt('e',119,105,'Add',colour='model',bold=True)
txt('e',121,201,'Graph hybrid prediction',colour='model',bold=True)
txt('e',121,223,'Weight set in training folds',colour='muted')
txt('e',16,255,'2  Fixed HFGE ensemble',bold=True,anchor='start')
txt('e',58,331,'Context-TabPFN')
txt('e',190,317,'Graph hybrid\nprediction',colour='model')
txt('e',122,307,'50:50',bold=True,colour='model')
txt('e',121,354,'Final HFGE prediction',bold=True)
txt('e',68,371,'London',bold=True);txt('e',178,371,'Manchester',bold=True)
txt('e',121,459,'Out-of-fold spatial predictions',colour='muted')

txt(None,436.68,1070,'Five contiguous spatial folds  •  Training-fold-only fitting  •  Out-of-fold evaluation',colour='muted')
assert original_paths==[E.tostring(p) for p in root.xpath('//s:path',namespaces=N)]
r.write(str(OUT/'figure3_original_repaired.svg'),encoding='utf-8',xml_declaration=True)
collisions=[]
for i,a in enumerate(labels):
 for b in labels[i+1:]:
  x,y,z,q=a['bbox'];u,v,w,t=b['bbox']
  if min(z,w)-max(x,u)>1 and min(q,t)-max(y,v)>1:collisions.append([a['text'],b['text']])
(OUT/'manifest.json').write_text(json.dumps({'all_original_graphical_paths_retained':True,'minimum_font_main_pt':14.5*478.8/873.36,'minimum_font_response_pt':14.5*435.6/873.36,'label_overlaps':collisions,'labels':labels},indent=2))
print(json.dumps({'labels':len(labels),'overlaps':collisions}))
