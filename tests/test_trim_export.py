import json
import subprocess
from test_app import client, wait


def test_trim_exports_synchronized_video_and_original_audio_without_stems(client,tmp_path):
    clip=tmp_path/'source.mp4'
    subprocess.run(['ffmpeg','-v','error','-y','-f','lavfi','-i','testsrc2=s=64x64:r=10:d=3',
                    '-f','lavfi','-i','sine=frequency=440:duration=3','-c:v','libx264','-c:a','aac','-shortest',str(clip)],check=True)
    p=client.post('/api/projects').json();pid=p['project_id'];h={'Authorization':'Bearer '+p['access_token']}
    with clip.open('rb') as f: assert client.post(f'/api/projects/{pid}/upload',headers=h,files={'file':('clip.mp4',f)}).status_code==200
    before=client.get(f'/api/projects/{pid}/original',headers=h).content
    url=f'/api/projects/{pid}/export'
    assert client.post(url,headers=h,json={'format':'mp4','include_video':True,'trim':{'start':2,'end':1}}).status_code==422
    assert client.post(url,headers=h,json={'format':'mp4','include_video':True,'trim':{'start':1,'end':4}}).status_code==422
    r=client.post(url,headers=h,json={'format':'mp4','include_video':True,'trim':{'start':1,'end':2}})
    assert r.status_code==202,r.text
    job=wait(client,'/api/exports/'+r.json()['export_id'],h);assert job['status']=='completed',job
    out=tmp_path/'trimmed.mp4';out.write_bytes(client.get(job['download_url'],headers=h).content)
    info=json.loads(subprocess.check_output(['ffprobe','-v','error','-show_streams','-of','json',str(out)]))
    video=next(s for s in info['streams'] if s['codec_type']=='video');audio=next(s for s in info['streams'] if s['codec_type']=='audio')
    assert int(video['nb_frames'])==10
    assert abs(float(video['duration'])-1)<.11 and abs(float(audio['duration'])-1)<.11
    import numpy as np
    samples=np.frombuffer(subprocess.check_output(['ffmpeg','-v','error','-i',str(out),'-f','f32le','-']),dtype='<f4')
    assert np.sqrt(np.mean(samples**2)) > .01
    wav=client.post(url,headers=h,json={'format':'wav','include_video':False,'trim':{'start':.5,'end':1.5}})
    assert wav.status_code==202
    wavjob=wait(client,'/api/exports/'+wav.json()['export_id'],h);assert wavjob['status']=='completed',wavjob
    wavefile=tmp_path/'trimmed.wav';wavefile.write_bytes(client.get(wavjob['download_url'],headers=h).content)
    wavinfo=json.loads(subprocess.check_output(['ffprobe','-v','error','-show_format','-of','json',str(wavefile)]))
    assert abs(float(wavinfo['format']['duration'])-1)<.01
    assert client.get(f'/api/projects/{pid}/original',headers=h).content==before
    assert client.get(f'/api/projects/{pid}/stems',headers=h).json()==[]


def test_trim_respects_muted_layers(client,tmp_path):
    import numpy as np
    clip=tmp_path/'tone.wav'
    subprocess.run(['ffmpeg','-v','error','-y','-f','lavfi','-i','sine=frequency=440:duration=3',str(clip)],check=True)
    p=client.post('/api/projects').json();pid=p['project_id'];h={'Authorization':'Bearer '+p['access_token']}
    with clip.open('rb') as f: assert client.post(f'/api/projects/{pid}/upload',headers=h,files={'file':('tone.wav',f)}).status_code==200
    r=client.post(f'/api/projects/{pid}/process',headers=h,json={'prompts':['tone']});r.raise_for_status()
    job=wait(client,'/api/jobs/'+r.json()['job_id'],h);assert job['status']=='completed'
    sid=job['stems'][0]['id'];assert client.patch(f'/api/projects/{pid}/stems/{sid}',headers=h,json={'muted':True}).status_code==200
    r=client.post(f'/api/projects/{pid}/export',headers=h,json={'format':'wav','include_video':False,'trim':{'start':.5,'end':1.5}});assert r.status_code==202
    job=wait(client,'/api/exports/'+r.json()['export_id'],h);assert job['status']=='completed',job
    out=tmp_path/'silent.wav';out.write_bytes(client.get(job['download_url'],headers=h).content)
    samples=np.frombuffer(subprocess.check_output(['ffmpeg','-v','error','-i',str(out),'-f','f32le','-']),dtype='<f4')
    assert len(samples)>0 and np.max(np.abs(samples))==0
