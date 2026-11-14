"""Plot joint test errors using the published result tables."""
import os,tempfile
from pathlib import Path
os.environ.setdefault('MPLCONFIGDIR',str(Path(tempfile.gettempdir())/'tempo-helm4-render-cache'))
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
import argparse,hashlib,json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.text import Text
from style import apply,size,REGIMES

HERE=Path(__file__).resolve().parent
CAPTION='Mean relative $L^2$ test errors.'
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--data',type=Path,default=HERE);parser.add_argument('--out',type=Path,default=HERE);args=parser.parse_args()
    args.out.mkdir(parents=True,exist_ok=True);apply()
    source=args.data/'joint_test_errors_values.json';data=json.loads(source.read_text())
    values=np.asarray(data['percent_values'],dtype=float);stds=np.asarray(data['sample_sd_percent'],dtype=float)
    assert values.shape==stds.shape==(4,len(data['tasks']))
    assert np.all(values[np.isfinite(values)]>0)
    assert np.all((stds[np.isfinite(stds)] >= 0))
    tasks=data['tasks'];names=data['display_names']
    labels=['Global POD','Global NeuralPOD','TEMPO(POD)','TEMPO(NeuralPOD)']
    fig,ax=plt.subplots(figsize=size(68));fig.subplots_adjust(left=.105,right=.99,bottom=.22,top=.82)
    x=np.arange(len(tasks));width=.18;error_bars=[]
    for i,(vals,sd,label,color) in enumerate(zip(values,stds,labels,[REGIMES[2],REGIMES[3],REGIMES[0],REGIMES[1]])):
        finite=np.isfinite(vals)
        ax.bar((x+(i-1.5)*width)[finite],vals[finite],width,label=label,color=color)
        error_bars.append([False] * len(tasks))
    ax.set_yscale('log');ax.set_ylim(.3,100);ax.set_ylabel('Relative $L^2$ error (%)',fontsize=9)
    ax.set_xticks(x,names);ax.tick_params(axis='x',length=0,pad=7);ax.tick_params(axis='both',labelsize=8.5)
    ax.set_yticks([.5,1,3,10,30,100],['0.5','1','3','10','30','100'])
    ax.spines[['top','right']].set_visible(False)
    ax.legend(ncol=4,loc='lower center',bbox_to_anchor=(.5,1.035),columnspacing=1.1,handlelength=1.,fontsize=8.5)
    fig.canvas.draw();renderer=fig.canvas.get_renderer()
    for item in fig.findobj(Text):
        if item.get_visible() and item.get_text():
            b=item.get_window_extent(renderer)
            if b.x0<-.5 or b.y0<-.5 or b.x1>fig.bbox.width+.5 or b.y1>fig.bbox.height+.5:raise ValueError('Text outside canvas: '+item.get_text())
    outputs={}
    for extension in ('pdf','png'):
        path=args.out/('joint_test_errors.'+extension);metadata={'CreationDate':None,'ModDate':None} if extension=='pdf' else None
        fig.savefig(path,dpi=300,facecolor='white',metadata=metadata)
        outputs[extension]=dict(file=path.name,bytes=path.stat().st_size,sha256=sha(path))
    plt.close(fig)
    proof=dict(status='rendered',numeric_input_sha256=sha(source),renderer_sha256=sha(__file__),style_sha256=sha(HERE/'style.py'),tasks=tasks,percent_values=data['percent_values'],sample_sd_percent=data['sample_sd_percent'],runs=data['runs'],error_bars_drawn=error_bars,model_checkpoints=data['model_checkpoints'],source_files=data['source_files'],cpu_threads=1,gpu_used=False,model_training=False,text_inside_canvas=True,outputs=outputs)
    (args.out/'joint_test_errors_renderer_proof.json').write_text(json.dumps(proof,indent=2)+'\n')
    (args.out/'joint_test_errors_caption.txt').write_text(CAPTION+'\n')
    print(json.dumps(dict(status='rendered',tasks=tasks)),flush=True)
if __name__=='__main__':main()
