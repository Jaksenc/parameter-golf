"""Execution-only recovery: same A/C/D/E/F cases and scorer, no generation.
Use only if the separately recorded bounded-generation run exhausts its deadline.
No benchmark labels or prompts are changed. Progress is persisted after every case.
"""
import argparse,json
from pathlib import Path
from semantic_train_v1 import Runtime,save,emit
from semantic_data_v1 import dataset,TRUTH,ASTS,OPS,evaluate,tests

def main():
    p=argparse.ArgumentParser();p.add_argument('--out',required=True);a=p.parse_args();out=Path(a.out);out.mkdir(exist_ok=False,parents=True)
    rows=[r for r in dataset() if r['split']=='development'];rt=Runtime();records=[]
    for row in rows:
        ps={m:rt.predict(row,m) for m in ['direct','fact_p','fact_q','rule']}
        if any(x['choice'] is None for x in ps.values()):raise RuntimeError('Ambiguous stage prediction; preserve failure, do not guess unknown')
        pf={atom:TRUTH[ps['fact_'+atom]['choice']] for atom in ['p','q']};pa=ASTS[ps['rule']['choice']]
        truths={'C_gold':row['truth'],'D_facts':evaluate(row['ast'],pf),'E_rule':evaluate(pa,row['facts']),'F_pipeline':evaluate(pa,pf)}
        rec={'id':row['id'],'operator':row['operator'],'gold_truth':row['truth'],'gold_facts':row['facts'],'gold_rule':OPS.index(row['operator']),'predictions':ps,'truths':truths,'A_correct':ps['direct']['choice']==row['answer']}
        records.append(rec);save(out/'partial_cases.json',records);emit('diagnostic_case',**rec)
    metrics={'n':len(rows),'A_direct':sum(r['A_correct'] for r in records),'B_reasoning_correct':None,'B_reasoning_n':0,'B_direct_same_subset':None,'fact_joint':sum(all(TRUTH[r['predictions']['fact_'+a]['choice']]==r['gold_facts'][a] for a in ['p','q']) for r in records),'rule_exact':sum(r['predictions']['rule']['choice']==r['gold_rule'] for r in records)}
    for k in ['C_gold','D_facts','E_rule','F_pipeline']:metrics[k]=sum(r['truths'][k]==r['gold_truth'] for r in records)
    stage='fact' if metrics['D_facts']<metrics['E_rule'] else ('rule' if metrics['E_rule']<metrics['D_facts'] else ('execute' if metrics['A_direct']<metrics['D_facts'] else 'rule'))
    result={'mode':'diagnostic_recovery_without_generation','metrics':metrics,'selected_stage':stage,'cases':records,'reasoning':[],'runtime':rt.receipt,'privileged_arms':['C_gold','D_facts','E_rule'],'data':tests(),'qualification':'Generation omitted only in recovery; original attempted reasoning run remains separate. Same non-generation cases, prompts and stage-selection rule.'}
    save(out/'diagnostic.json',result);emit('diagnostic_complete',metrics=metrics,selected_stage=stage)
if __name__=='__main__':main()
