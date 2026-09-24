"""Durable resume ledger. Never repeats a successful call or erases a failure."""
from __future__ import annotations
import json,sqlite3,time
from pathlib import Path
from .cases import digest

class PriorFailure(RuntimeError):pass
class Ledger:
    def __init__(self,path,identity):
      self.path=Path(path);self.path.parent.mkdir(parents=True,exist_ok=True)
      self.db=sqlite3.connect(self.path);self.db.execute('PRAGMA journal_mode=WAL');self.db.execute('PRAGMA synchronous=FULL')
      self.db.executescript('CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY,value TEXT NOT NULL);CREATE TABLE IF NOT EXISTS calls (key TEXT PRIMARY KEY,payload TEXT NOT NULL,hash TEXT NOT NULL);CREATE TABLE IF NOT EXISTS attempts (seq INTEGER PRIMARY KEY AUTOINCREMENT,key TEXT NOT NULL,status TEXT NOT NULL,payload TEXT NOT NULL,stamp REAL NOT NULL);')
      current=self.db.execute('SELECT value FROM metadata WHERE key="identity"').fetchone();value=digest(identity)
      if current and current[0]!=value:raise ValueError('Ledger experiment/model identity differs')
      self.db.execute('INSERT OR IGNORE INTO metadata VALUES ("identity",?)',(value,));self.db.commit()
    def get(self,key):
      r=self.db.execute('SELECT payload,hash FROM calls WHERE key=?',(key,)).fetchone()
      if not r:return None
      x=json.loads(r[0])
      if digest(x)!=r[1]:raise RuntimeError('Corrupt cached call')
      return x
    def call(self,key,fn,retry_errors=False):
      old=self.get(key)
      if old is not None:return old
      failed=self.db.execute('SELECT 1 FROM attempts WHERE key=? AND status="error"',(key,)).fetchone()
      if failed and not retry_errors:raise PriorFailure('Prior failed call; explicit --retry-errors required')
      try:
        payload=fn();text=json.dumps(payload,sort_keys=True,allow_nan=False);fingerprint=digest(payload)
      except Exception as e:
        self.db.execute('INSERT INTO attempts(key,status,payload,stamp) VALUES (?,?,?,?)',(key,'error',json.dumps({'type':type(e).__name__,'message':str(e)}),time.time()));self.db.commit();raise
      with self.db:
        self.db.execute('INSERT INTO calls VALUES (?,?,?)',(key,text,fingerprint))
        self.db.execute('INSERT INTO attempts(key,status,payload,stamp) VALUES (?,?,?,?)',(key,'success',text,time.time()))
      return payload
    def receipt(self):
      return {'successful_calls':self.db.execute('SELECT count(*) FROM calls').fetchone()[0],
              'failed_attempts':self.db.execute('SELECT count(*) FROM attempts WHERE status="error"').fetchone()[0]}
    def close(self):self.db.close()


def verify_saved_call(connection,key,payload):
    row=connection.execute('SELECT payload,hash FROM calls WHERE key=?',(key,)).fetchone()
    if row is None:raise ValueError('Output is not backed by a saved successful call')
    saved=json.loads(row[0])
    if digest(saved)!=row[1] or saved!=payload:raise ValueError('Saved call and result differ')
