"""Readout-v2 entry; preserve original failed preflight instead of overwriting it."""
from pathlib import Path
import argparse,hashlib,zipfile
import remote_stage1 as r
import interface_v2
original_setup=r.setup_source
original_save=r.save
HERE=Path(__file__).parent

def setup():
    root,h=original_setup();interface_v2.apply(root);return root,h

def save(name,obj,publish=True):
    obj['readout_interface']=2
    obj['revised_core_sha256']='946de0d3aa1ae65a547fe0e6efa700ac267d9ffc3636d38bf739d39b6006e3c9'
    obj['interface_source_sha256']=hashlib.sha256((HERE/'interface_v2.py').read_bytes()).hexdigest()
    return original_save('v2-'+name,obj,publish)

r.setup_source=setup
r.save=save
r.__file__=__file__
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--preflight',action='store_true');p.add_argument('--worker',action='store_true');p.add_argument('--input');p.add_argument('--output');p.add_argument('--shard',type=int,default=0);a=p.parse_args()
    if a.preflight:
        try:r.preflight()
        finally:
            if Path('shared-source.zip').exists():
                with zipfile.ZipFile('shared-source.zip','a',zipfile.ZIP_DEFLATED) as z:
                    for n in ('entry_v2.py','interface_v2.py'):z.write(HERE/n,'research/'+n)
    elif a.worker:r.worker(a.input,a.output)
    else:
        if not 0<=a.shard<6:raise ValueError('shard')
        r.run(a.shard)
