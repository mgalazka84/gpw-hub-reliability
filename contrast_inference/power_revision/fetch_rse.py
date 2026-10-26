"""Retrieve the original public research dataset without redistributing records."""
from pathlib import Path
import hashlib,urllib.request
URL='https://openpsychometrics.org/_rawdata/RSE.zip'
SHA256='876f95a1db34c8fe0e41a352c32edb0118c959a6675669fc0b3e1aba2271e893'
def fetch():
 p=Path(__file__).parent/'data'/'RSE.zip';p.parent.mkdir(exist_ok=True)
 if not p.exists():
  with urllib.request.urlopen(URL,timeout=60) as r:b=r.read()
  if hashlib.sha256(b).hexdigest()!=SHA256:raise RuntimeError('Source bytes changed; inspect provenance before analysing')
  p.write_bytes(b)
 if hashlib.sha256(p.read_bytes()).hexdigest()!=SHA256:raise RuntimeError('Local input hash mismatch')
 print('Verified original RSE archive',len(p.read_bytes()),'bytes')
if __name__=='__main__':fetch()
