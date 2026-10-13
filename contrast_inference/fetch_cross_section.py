"""Fetch original UCI data under CC BY 4.0 and record byte hashes/provenance."""
from pathlib import Path
import urllib.request,hashlib,json,zipfile,io

ROOT=Path(__file__).parent/'data'/'cross_section'

def run():
    ROOT.mkdir(parents=True,exist_ok=True);records=[]
    for name,number,doi in [('abalone',1,'10.24432/C55C7W'),('dry_bean',602,'10.24432/C50S4B')]:
        slug='abalone' if name=='abalone' else 'dry%2Bbean%2Bdataset'
        url=f'https://archive.ics.uci.edu/static/public/{number}/{slug}.zip'
        path=ROOT/(name+'.zip')
        if not path.exists():path.write_bytes(urllib.request.urlopen(url,timeout=40).read())
        archive=zipfile.ZipFile(io.BytesIO(path.read_bytes()))
        keep=[f for f in archive.namelist() if f.endswith(('.data','.names','.xlsx'))]
        extracted=[]
        for f in keep:
            target=ROOT/Path(f).name;data=archive.read(f);target.write_bytes(data)
            extracted.append(dict(file=target.name,size=len(data),sha256=hashlib.sha256(data).hexdigest()))
        records.append(dict(dataset=name,url=url,doi=doi,license='CC BY 4.0',
            archive_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),files=extracted))
    (ROOT/'PROVENANCE.json').write_text(json.dumps(records,indent=2))
    (ROOT/'LICENSE_DATA.txt').write_text('These UCI datasets are redistributed under Creative Commons Attribution 4.0.\nAbalone: Nash, Sellers, Talbot, Cawthorn and Ford (1994), doi:10.24432/C55C7W.\nDry Bean: Koklu and Ozkan (2020), doi:10.24432/C50S4B.\nLicense: https://creativecommons.org/licenses/by/4.0/\nOriginal measurement files are unchanged. Derived files and analyses identify all transformations.\n')
    print(json.dumps(records,indent=2),flush=True)

if __name__=='__main__':run()
