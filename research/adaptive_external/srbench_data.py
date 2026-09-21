"""Public data acquisition only; no solver, test scoring, or private project files."""
import base64, concurrent.futures, csv, gzip, hashlib, io, json, math, random, urllib.request
from pathlib import Path
from bbeh_run import publish, sha
REF='7c1f4bdc00136dc2e55c87fa6b8ba6e8af6d1a68'
NAMES=['1028_SWD','1089_USCrime','1193_BNG_lowbwt','1199_BNG_echoMonths','192_vineyard','210_cloud','522_pm10','557_analcatdata_apnea1','579_fri_c0_250_5','606_fri_c2_1000_10','650_fri_c0_500_50','678_visualizing_environmental','first_principles_absorption','first_principles_bode','first_principles_hubble','first_principles_ideal_gas','first_principles_kepler','first_principles_leavitt','first_principles_newton','first_principles_planck','first_principles_rydberg','first_principles_schechter','first_principles_supernovae_zr','first_principles_tully_fisher']
def acquire(name):
    path=f'datasets/{name}/{name}.tsv.gz';url=f'https://raw.githubusercontent.com/EpistasisLab/pmlb/{REF}/{path}'
    report={'dataset':name,'source_url':url}
    try:
        raw=urllib.request.urlopen(url,timeout=60).read(50_000_001)
        if raw.startswith(b'version https://git-lfs.github.com/spec/v1'):
            pointer=raw.decode();expected=[x.split('sha256:')[1] for x in pointer.splitlines() if x.startswith('oid ')][0]
            url=f'https://media.githubusercontent.com/media/EpistasisLab/pmlb/{REF}/{path}'
            raw=urllib.request.urlopen(url,timeout=60).read(50_000_001)
            if sha(raw)!=expected:raise ValueError('LFS content checksum mismatch')
            report['lfs_sha256']=expected
        if len(raw)>50_000_000:raise ValueError('Dataset compressed size limit exceeded')
        text=gzip.decompress(raw).decode('utf-8');reader=csv.DictReader(io.StringIO(text),delimiter='\t');fields=reader.fieldnames
        if fields is None or 'target' not in fields:raise ValueError('Missing target column')
        features=[f for f in fields if f!='target'];rows=list(reader);report.update(source_sha256=sha(raw),uncompressed_sha256=sha(text.encode()),n_rows=len(rows),n_features=len(features),features=features)
        if not 1<=len(features)<=2:report['status']='unsupported_dimensions';return report,None
        if len(rows)<20:report['status']='insufficient_rows';return report,None
        selected=list(range(len(rows)));random.Random(20260920).shuffle(selected);selected=sorted(selected[:256])
        array=[[float(rows[i][f]) for f in features+['target']] for i in selected]
        if not all(math.isfinite(v) for row in array for v in row):raise ValueError('Nonfinite selected values')
        data={'dataset':name,'features':features,'original_row_ids':selected,'values':array}
        report.update(status='included',selected_rows=len(selected),selection='all if n<=256; otherwise fixed Random(20260920) row sample independent of values',selected_sha256=sha(json.dumps(data,sort_keys=True,separators=(',',':')).encode()))
        return report,data
    except Exception as exc:report.update(status='acquisition_error',error=f'{type(exc).__name__}: {exc}');return report,None
def main():
    reports=[];datasets=[]
    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
        for report,data in pool.map(acquire,NAMES):
            reports.append(report)
            if data:datasets.append(data)
            print('DATASET '+json.dumps(report),flush=True)
    obj={'pmlb_commit':REF,'selection_rule':'all 24 official SRBench2025 datasets with 1 or 2 numerical predictors and at least 20 rows; <=256 label-independently selected rows; not canonical full SRBench','datasets':datasets}
    packed=gzip.compress(json.dumps(obj,separators=(',',':')).encode(),mtime=0)
    manifest={'protocol':obj['selection_rule'],'pmlb_commit':REF,'reports':reports,'pack_sha256':sha(packed),'decoded_sha256':sha(json.dumps(obj,separators=(',',':')).encode()),'pack_bytes':len(packed),'dataset_count':len(datasets),'no_benchmark_scores_computed':True}
    print(publish('research/adaptive_external/results/srbench-data-manifest.json',json.dumps(manifest,indent=2)),flush=True)
    print(publish('research/adaptive_external/results/srbench-data-pack.json',json.dumps({'encoding':'base64(gzip(json))','sha256':sha(packed),'data':base64.b64encode(packed).decode()})),flush=True)
if __name__=='__main__':main()
