import copy,json,tempfile,unittest
from pathlib import Path
import recover_v8 as r
import countercase_v8 as c


def setup(root,missing=()):
    jobs=[[] for _ in range(32)]
    for i in range(295):
        row={'id':str(i),'state':'fictional software fixture','question':{'type':'choice','instructions':'fixture','criteria':{'no':'no','yes':'yes'}},'labels':['no','yes']}
        jobs[i%32].append({'input':row,'arms':['countercase480']})
    p=root/'countercase-prepared';p.mkdir()
    c.prior.h5.write(p/'jobs.json',jobs)
    c.prior.h5.write(p/'manifest.json',{'source_sha256':r.SOURCE,'jobs_hash':c.prior.h5.digest(jobs)})
    for i,plan in enumerate(jobs):
        d=root/'original-records'/f'countercase-v8-shard-{i}';d.mkdir(parents=True)
        c.prior.h5.write(d/'preflight.json',{'source_sha256':r.SOURCE})
        rows=[]
        for j in plan:
            if j['input']['id'] in missing:continue
            sig=c.prior.h5.digest(j['input'])
            rows.append({'id':j['input']['id'],'input_sha256':sig,'outputs':{'countercase480':{'arm':'countercase480','input_sha256':sig,'answer':'yes'}},'execution_order':['countercase480'],'shard':i})
        p=d/'records.jsonl';p.write_text(''.join(json.dumps(x)+'\n' for x in rows))
    return jobs

class RecoveryTests(unittest.TestCase):
    def test_all_preserved(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);setup(root)
            rec=r.prepare(root)
            self.assertEqual((rec['original_count'],rec['missing_count']),(295,0))
    def test_exact_missing_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);setup(root,missing={'2','19','70'})
            rec=r.prepare(root)
            jobs=json.loads((root/'recovery-prepared/jobs.json').read_text())
            self.assertEqual({j['input']['id'] for group in jobs for j in group},{'2','19','70'})
            self.assertEqual(rec['original_count'],292)
    def test_trailing_uncommitted_line_not_reused(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);setup(root)
            p=root/'original-records/countercase-v8-shard-0/records.jsonl'
            p.write_bytes(p.read_bytes().rstrip(b'\n'))
            rec=r.prepare(root)
            self.assertEqual(rec['missing_count'],1)
            self.assertGreater(rec['shard_audits'][0]['trailing_uncommitted_bytes'],0)
    def test_wrong_source_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);setup(root)
            p=root/'original-records/countercase-v8-shard-1/preflight.json';p.write_text('{"source_sha256":"changed"}')
            with self.assertRaises(ValueError):r.prepare(root)
    def test_output_choices_do_not_change_missing_selection(self):
        results=[]
        with tempfile.TemporaryDirectory() as tmp:
            for k in range(2):
                root=Path(tmp)/str(k);root.mkdir();setup(root,missing={'2','19'})
                if k:
                    for p in (root/'original-records').glob('*/records.jsonl'):
                        rows=[json.loads(s) for s in p.read_text().splitlines()]
                        for row in rows:row['outputs']['countercase480']['answer']='no'
                        p.write_text(''.join(json.dumps(x)+'\n' for x in rows))
                r.prepare(root);results.append(json.loads((root/'recovery-prepared/jobs.json').read_text()))
        self.assertEqual(results[0],results[1])
    def test_illegal_label_stops_not_discards(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);setup(root)
            p=root/'original-records/countercase-v8-shard-0/records.jsonl';rows=[json.loads(s) for s in p.read_text().splitlines()]
            rows[0]['outputs']['countercase480']['answer']='not in allowed labels';p.write_text(''.join(json.dumps(x)+'\n' for x in rows))
            with self.assertRaises(ValueError):r.prepare(root)
    def test_duplicate_stops(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);setup(root)
            p=root/'original-records/countercase-v8-shard-0/records.jsonl';s=p.read_text();p.write_text(s+s.splitlines()[0]+'\n')
            with self.assertRaises(ValueError):r.prepare(root)
    def test_aggregate_complete_preserves(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);setup(root,missing={'2'})
            r.prepare(root);jobs=json.loads((root/'recovery-prepared/jobs.json').read_text())
            original=json.loads((root/'recovery-prepared/original.json').read_text())
            for i,plan in enumerate(jobs):
                d=root/'recovered-records'/f'countercase-v8-recovered-{i}';d.mkdir(parents=True)
                rr=[]
                for j in plan:
                    sig=c.prior.h5.digest(j['input'])
                    rr.append({'id':j['input']['id'],'input_sha256':sig,'outputs':{'countercase480':{'arm':'countercase480','input_sha256':sig,'answer':'no'}}})
                p=d/'records.jsonl';p.write_text(''.join(json.dumps(x)+'\n' for x in rr))
                c.prior.h5.write(d/'complete.json',{'source_sha256':r.SOURCE,'jobs_hash':c.prior.h5.digest(plan),'records_sha256':c.prior.h5.filehash(p),'count':len(rr)})
            r.aggregate(root)
            allrows=json.loads((root/'all_records.json').read_text());self.assertEqual(len(allrows),295)
            self.assertTrue(all(row in allrows for row in original))

if __name__=='__main__':unittest.main()
