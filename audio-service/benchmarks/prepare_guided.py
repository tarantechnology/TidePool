"""Fetch small attributed recordings and build a fixed, label-free smoke benchmark.

Requires an existing LibriSpeech dummy parquet via --speech-parquet (pyarrow extra).
Generated sources are not committed. The manifest records source and derivative hashes.
"""
import argparse
import csv
import hashlib
import io
import json
import urllib.request
from pathlib import Path
import pyarrow.parquet as pq
import soundfile as sf


def fetch(url,path):
    if not path.exists():
        with urllib.request.urlopen(url,timeout=60) as response: path.write_bytes(response.read())
    return path


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--speech-parquet',type=Path,required=True);p.add_argument('--out',type=Path,required=True);args=p.parse_args()
    root=args.out;root.mkdir(parents=True,exist_ok=True)
    sources={}
    def source(key,path,url,license,attribution,**extra):
        sources[key]={'file':path.name,'url':url,'license':license,'attribution':attribution,'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),**extra}
    index=json.loads(fetch('https://raw.githubusercontent.com/librosa/librosa/main/librosa/util/example_data/index.json',root/'librosa-index.json').read_text())
    for key in ('brahms','nutcracker','libri1','libri2'):
        base=index[key]['path'];url='https://librosa.org/data/audio/'+base
        path=fetch(url+'.ogg',root/(key+'.ogg'))
        license_text=fetch(url+'.txt',root/(key+'-license.txt')).read_text()
        source(key,path,url+'.ogg','See included '+key+'-license.txt',license_text)
    licenses=fetch('https://raw.githubusercontent.com/karolpiczak/ESC-50/master/LICENSE',root/'ESC-LICENSE.txt').read_text()
    table=list(csv.DictReader(io.StringIO(fetch('https://raw.githubusercontent.com/karolpiczak/ESC-50/master/meta/esc50.csv',root/'esc50.csv').read_text())))
    for category in ('dog','rain'):
        for fold in ('1','2'):
            row=next(r for r in table if r['category']==category and r['fold']==fold and r['esc10']=='True')
            name=row['filename'];url='https://raw.githubusercontent.com/karolpiczak/ESC-50/master/audio/'+name
            attribution=next(line for line in licenses.splitlines() if name.rsplit('-',1)[0]+'.ogg' in line)
            source(category+fold,fetch(url,root/name),url,'ESC-10 CC BY 3.0; individual license in attribution',attribution,source_recording=row['src_file'])
    speech=[]
    for row in pq.read_table(args.speech_parquet,columns=['audio','id']).to_pylist():
        data,rate=sf.read(io.BytesIO(row['audio']['bytes']))
        if len(data)<6*rate:continue
        path=root/(row['id']+'.wav');sf.write(path,data,rate,subtype='FLOAT');key='speech'+str(len(speech));speech.append(key)
        source(key,path,'https://huggingface.co/datasets/hf-internal-testing/librispeech_asr_dummy','CC BY 4.0','LibriSpeech: Panayotov, Chen, Povey, Khudanpur; source '+row['id'],source_recording=row['id'])
        if len(speech)==2:break
    if len(speech)<2:raise ValueError('Need two existing speech recordings')
    examples=[]
    for split,voice,other,music,fold in [('tuning','speech0','speech1','brahms','1'),('heldout','libri1','libri2','nutcracker','2')]:
        for index,(category,target,bg,prompt) in enumerate([
            ('speech_music',voice,music,'person speaking'),('speech_noise',voice,'rain'+fold,'person speaking'),
            ('intermittent','dog'+fold,music,'dog barking'),('competing_voices',voice,other,'man speaking'),('stereo',voice,music,'person speaking')]):
            item={'id':split+'-'+category,'split':split,'category':category,'seed':1729+index,'seconds':12,
                  'target':sources[target],'background':sources[bg],'prompt':prompt,'anchors':[{'token':'+','start':1,'end':3}]}
            if category=='intermittent':item['placements']=[[1,3],[6,8],[10,11]]
            examples.append(item)
    manifest={'version':1,'note':'Small smoke corpus, one held-out example per category. Short sources are repeated with deterministic offsets. Competing voices include same-speaker overlaps in tuning. No manual labels or training. Not sufficient for profile promotion.','examples':examples}
    (root/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')

if __name__=='__main__':main()
