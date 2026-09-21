"""One verified lexer repair; prompts, caps, selection and scoring remain unchanged.
Original run35550920040 failed unit tests before any target-data inference.
"""
from __future__ import annotations
import hashlib,json,re,sys
from pathlib import Path
import study
OLD=r'(?<![\w.])[-+]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?%?(?![\w.])'
NEW=r'(?<![\w.])[-+]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?%?(?!\w)(?!\.\d)'
if study.NUM.pattern!=OLD:raise RuntimeError('Unexpected original lexer; do not patch different source')
study.NUM=re.compile(NEW)
for text,value in [('25.','25'),('12.5%.','1/8'),('(300).','-300'),('1,250.','1250'),('1.234%.','617/50000')]:
 bank=study.number_bank(text,'?')
 if len(bank)!=1 or next(iter(bank.values()))['value']!=value:raise RuntimeError('Numeric boundary regression')

def main():
 args=sys.argv[1:]
 if not args:raise ValueError('choose test, prepare, run or audit')
 mode=args.pop(0)
 if mode=='test':
  import unittest,test_study
  result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromModule(test_study))
  if not result.wasSuccessful():raise SystemExit(1)
 elif mode=='prepare':
  study.prepare('data');p=Path('data/manifest.json');m=json.loads(p.read_text());m['lexer_repair']={'old':OLD,'new':NEW,'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'before_target_inference':True};study.save(p,m)
 elif mode=='run':
  import run
  sys.argv=['run.py']+args;run.main()
 elif mode=='audit':
  import audit
  sys.argv=['audit.py']+args;audit.main()
 else:raise ValueError(mode)
if __name__=='__main__':main()
