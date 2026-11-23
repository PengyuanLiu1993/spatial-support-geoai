"""Rearrange the original five vector panels without redrawing their contents."""
from pathlib import Path
from copy import deepcopy
from io import BytesIO
import hashlib
import json
from lxml import etree as ET
from pypdf import PdfReader, PdfWriter, PageObject, Transformation
from pypdf.generic import RectangleObject
from reportlab.pdfgen import canvas
import pypdfium2 as pdfium

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'assets/framework/original'
OUT = ROOT / 'results/revision_20260925/figures'
SVG = 'http://www.w3.org/2000/svg'
NS = {'s': SVG}
WIDTH, HEIGHT = 873.36, 1084.0
OLD_HEIGHT = 577.412977
Y, PANEL_HEIGHT = 50.298645, 477.792
# a,b,c in the first row; d,e centred in the second row.
PANELS = [
    ('a', 'axes_1', 'patch_2', 5.76, 181.44, 12.0, 44.0),
    ('b', 'axes_4', 'patch_3', 211.392, 325.08, 217.44, 44.0),
    ('c', 'axes_6', 'patch_4', 560.664, 294.84, 566.52, 44.0),
    ('d', 'axes_7', 'patch_5', 879.696, 325.08, 140.424, 565.0),
    ('e', 'axes_8', 'patch_6', 1228.968, 243.432, 489.504, 565.0),
]
ARROWS = [
    [(196, 282.896), (213, 282.896)],
    [(545.5, 282.896), (562.5, 282.896)],
    [(713.94, 525.8), (713.94, 543), (302.964, 543), (302.964, 561)],
    [(469.5, 803.896), (485.5, 803.896)],
]

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    original = ET.parse(str(SOURCE.with_suffix('.svg')))
    root = deepcopy(original.getroot())
    root.set('width', f'{WIDTH}pt'); root.set('height', f'{HEIGHT}pt')
    root.set('viewBox', f'0 0 {WIDTH} {HEIGHT}')
    old = original.xpath('.//s:g[@id="figure_1"]', namespaces=NS)[0]
    old_by_id = {e.get('id'): e for e in old}
    root.remove(root.xpath('./s:g[@id="figure_1"]', namespaces=NS)[0])
    figure = ET.SubElement(root, f'{{{SVG}}}g', id='figure_1')
    ET.SubElement(figure, f'{{{SVG}}}rect', x='0', y='0', width=str(WIDTH), height=str(HEIGHT), fill='white')
    checks = []
    for letter, axis, frame, x, width, nx, ny in PANELS:
        group = ET.SubElement(figure, f'{{{SVG}}}g', id=f'panel_{letter}', transform=f'translate({nx-x} {ny-Y})')
        for key in [frame, axis]:
            source = old_by_id[key]
            copy = deepcopy(source)
            group.append(copy)
            assert ET.tostring(copy) == ET.tostring(source)
        checks.append({'panel': letter, 'content_unchanged': True, 'x': nx, 'y': ny})
    for identifier, tx, ty in [('text_83', WIDTH/2-739.08, 0), ('text_84', WIDTH/2-739.08, HEIGHT-14-569.610984)]:
        g = ET.SubElement(figure, f'{{{SVG}}}g', transform=f'translate({tx} {ty})')
        g.append(deepcopy(old_by_id[identifier]))
    for points in ARROWS:
        ET.SubElement(figure, f'{{{SVG}}}polyline', points=' '.join(f'{x},{y}' for x,y in points), fill='none', stroke='#1c2532', **{'stroke-width':'1.2'})
        x,y=points[-1]; px,py=points[-2]
        tri=[(x,y),(x-6,y-3),(x-6,y+3)] if x>px else [(x,y),(x-3,y-6),(x+3,y-6)]
        ET.SubElement(figure, f'{{{SVG}}}polygon', points=' '.join(f'{a},{b}' for a,b in tri), fill='#1c2532')
    target = OUT / 'figure2_hfge_revised'
    ET.ElementTree(root).write(str(target.with_suffix('.svg')), encoding='utf-8', xml_declaration=True)

    # Preserve original vector paths and embedded fonts in the PDF as well.
    page = PageObject.create_blank_page(width=WIDTH, height=HEIGHT)
    source_pdf = PdfReader(SOURCE.with_suffix('.pdf')).pages[0]
    def place(rect, nx, ny):
        x,y,w,h=rect
        piece=deepcopy(source_pdf)
        piece.cropbox=RectangleObject([x, OLD_HEIGHT-y-h, x+w, OLD_HEIGHT-y])
        piece.trimbox=deepcopy(piece.cropbox)
        trans=Transformation().translate(nx-x, HEIGHT-ny-(OLD_HEIGHT-y))
        page.merge_transformed_page(piece, trans, over=True, expand=False)
    for _,_,_,x,w,nx,ny in PANELS:
        place((x-1,Y-1,w+2,PANEL_HEIGHT+2),nx-1,ny-1)
    # Original heading/footer, centred on the new canvas.
    place((400,0,679,30), WIDTH/2-339.5, 0)
    place((380,554,720,23.413), WIDTH/2-360, HEIGHT-29.611)
    overlay=BytesIO(); cv=canvas.Canvas(overlay,pagesize=(WIDTH,HEIGHT))
    cv.setStrokeColorRGB(28/255,37/255,50/255); cv.setFillColorRGB(28/255,37/255,50/255); cv.setLineWidth(1.2)
    for points in ARROWS:
        p=cv.beginPath();p.moveTo(points[0][0],HEIGHT-points[0][1])
        for x,y in points[1:]:p.lineTo(x,HEIGHT-y)
        cv.drawPath(p)
        x,y=points[-1];px,py=points[-2]
        tri=[(x,y),(x-6,y-3),(x-6,y+3)] if x>px else [(x,y),(x-3,y-6),(x+3,y-6)]
        p=cv.beginPath();p.moveTo(tri[0][0],HEIGHT-tri[0][1])
        for x,y in tri[1:]:p.lineTo(x,HEIGHT-y)
        p.close();cv.drawPath(p,stroke=0,fill=1)
    cv.showPage();cv.save();overlay.seek(0)
    page.merge_page(PdfReader(overlay).pages[0])
    writer=PdfWriter();writer.add_page(page)
    with target.with_suffix('.pdf').open('wb') as f:writer.write(f)
    pdf=pdfium.PdfDocument(target.with_suffix('.pdf'))
    pdf[0].render(scale=300/72).to_pil().save(target.with_suffix('.png'))
    pdf[0].render(scale=1600/WIDTH).to_pil().save(OUT/'figure2_hfge_preview.png')
    manifest={'source_svg':str(SOURCE.with_suffix('.svg').relative_to(ROOT)), 'source_sha256':hashlib.sha256(SOURCE.with_suffix('.svg').read_bytes()).hexdigest(), 'layout':'original five panels in two rows (a,b,c / d,e)', 'panel_contents':checks, 'effective_panel_and_text_scale_vs_original_at_same_width':1478.16/WIDTH}
    (OUT/'figure2_hfge_relayout.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    print('Preserved all five vector panels; created',target)

if __name__=='__main__':main()
