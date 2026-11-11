"""Fixed loss traces using existing ReportLab plotting and Poppler, no install."""
from pathlib import Path
import csv,json,math,subprocess,shutil
import numpy as np
import reportlab
from reportlab.graphics.shapes import Drawing,String,Line,Rect
from reportlab.graphics.charts.lineplots import LinePlot
from reportlab.graphics import renderPDF,renderSVG
from reportlab.lib.colors import HexColor,black,white
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'outputs/phase216/phase216_20261023'
def main():
    traces={a:[json.loads(x) for x in (OUT/f'delivery/{a}_LOSSES.jsonl').read_text(encoding='utf-8').splitlines()] for a in ['C','D']}
    assert all(len(t)==4000 for t in traces.values());rows=[]
    for a,trace in traces.items():rows.extend({'arm':a,'step':r['step'],'target_loss':r['target_loss'],'source_KL':r['source_KL']} for r in trace)
    with (OUT/'LOSS_TRACE.csv').open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    profile={a:{k:{'n':4000,'missing':0,'min':float(np.min([r[k] for r in t])),'median':float(np.median([r[k] for r in t])),'max':float(np.max([r[k] for r in t]))} for k in ['target_loss','source_KL']} for a,t in traces.items()}
    (OUT/'LOSS_PROFILE.json').write_text(json.dumps({'profile':profile,'chart':'two separate loss panels, every step and fixed100-step means','uncertainty':'none; serial iterations are not independent replications','backend':'existing ReportLab chart library; Matplotlib unavailable, no environment change'},indent=2),encoding='utf-8')
    font=Path(reportlab.__file__).parent/'fonts/Vera.ttf';pdfmetrics.registerFont(TTFont('Vera',str(font)))
    d=Drawing(518.4,374.4);d.add(Rect(0,0,518.4,374.4,fillColor=white,strokeColor=None))
    x=np.arange(1,4001);xb=x.reshape(40,100).mean(1);blue=HexColor('#0072B2');orange=HexColor('#D55E00')
    def text(x,y,s,size=9):d.add(String(x,y,s,fontName='Vera',fontSize=size,fillColor=black))
    def panel(y,title,arrays,styles,legend):
        p=LinePlot();p.x=53;p.y=y;p.width=439;p.height=99;p.data=arrays
        p.xValueAxis.valueMin=0;p.xValueAxis.valueMax=4000;p.xValueAxis.valueSteps=[0,1000,2000,3000,4000]
        p.xValueAxis.labels.fontName='Vera';p.xValueAxis.labels.fontSize=8
        ymax=max(max(v[1] for v in a) for a in arrays);unit=10**math.floor(math.log10(max(ymax,1e-10)))
        top=math.ceil(ymax/unit)*unit or 1
        p.yValueAxis.valueMin=min(0,min(v[1] for a in arrays for v in a));p.yValueAxis.valueMax=top;p.yValueAxis.valueSteps=[top*i/4 for i in range(5)]
        p.yValueAxis.labels.fontName='Vera';p.yValueAxis.labels.fontSize=8;p.yValueAxis.labelTextFormat='%g'
        p.xValueAxis.strokeWidth=.5;p.yValueAxis.strokeWidth=.5
        for i,(color,width,dash) in enumerate(styles):p.lines[i].strokeColor=color;p.lines[i].strokeWidth=width;p.lines[i].strokeDashArray=dash
        d.add(p);text(53,y+130,title,10)
        for i,(name,color,dash) in enumerate(legend):
            lx=53+i*220;d.add(Line(lx,y+114,lx+20,y+114,strokeColor=color,strokeWidth=1.5,strokeDashArray=dash));text(lx+25,y+111,name,8)
    target=[np.array([r['target_loss'] for r in traces[a]]) for a in ['C','D']]
    arrays=[list(zip(x.astype(float),v)) for v in target]+[list(zip(xb,v.reshape(40,100).mean(1))) for v in target]
    panel(222,'A. Original target supervision (loss)',arrays,[(HexColor('#bdd7e7'),.25,None),(HexColor('#f3c9ba'),.25,None),(blue,1.5,None),(orange,1.5,[5,3])],[('C: 100-step mean',blue,None),('D: 100-step mean',orange,[5,3])])
    kl=np.array([r['source_KL'] for r in traces['D']]);panel(62,'B. Source KL (lambda=1, T=1; loss)',[list(zip(x.astype(float),kl)),list(zip(xb,kl.reshape(40,100).mean(1)))],[(HexColor('#f3c9ba'),.25,None),(orange,1.5,[5,3])],[('D: 100-step mean',orange,[5,3])])
    text(112,30,'Optimizer update (single seed; no validation selection)',9)
    text(53,12,'Pale lines: every step. Bold lines: 100-step means. No uncertainty band.',8)
    renderPDF.drawToFile(d,str(OUT/'LOSS_CURVES.pdf'));renderSVG.drawToFile(d,str(OUT/'LOSS_CURVES.svg'))
    poppler=shutil.which('pdftoppm');assert poppler,'Use existing bundled Poppler only'
    subprocess.run([poppler,'-singlefile','-r','300','-png',str(OUT/'LOSS_CURVES.pdf'),str(OUT/'LOSS_CURVES')],check=True)
    from PIL import Image
    with Image.open(OUT/'LOSS_CURVES.png') as im:assert im.width>=2100 and im.height>=1500;im.convert('L').save(OUT/'LOSS_CURVES_GRAY.png')
    (OUT/'FIGURE_MACHINE_QA.json').write_text(json.dumps({'status':'machine_passed_visual_pending','size_inches':[7.2,5.2],'dpi':300,'embedded_TrueType_font':str(font),'minimum_text_points':8,'zero_baseline':True,'independent_y_scales':True,'data_rows':8000,'all_finite':all(np.isfinite(v).all() for v in target+[kl]),'backend':'existing ReportLab and Poppler','new_environment_changes':0},indent=2),encoding='utf-8')
    print(json.dumps({'figure':str(OUT/'LOSS_CURVES.png'),'rows':8000}))
if __name__=='__main__':main()
