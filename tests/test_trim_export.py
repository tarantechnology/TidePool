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


def test_clip_sequence_reorders_and_repeats_audio_and_video(client,tmp_path):
    import numpy as np
    clip=tmp_path/'sequence.mp4'
    subprocess.run(['ffmpeg','-v','error','-y','-f','lavfi','-i','testsrc2=s=64x64:r=10:d=3',
                    '-f','lavfi','-i',"sine=frequency=440:duration=1",'-f','lavfi','-i','sine=frequency=880:duration=1',
                    '-f','lavfi','-i','sine=frequency=1320:duration=1',
                    '-filter_complex','[1:a][2:a][3:a]concat=n=3:v=0:a=1[a]',
                    '-map','0:v','-map','[a]','-c:v','libx264','-c:a','aac',str(clip)],check=True)
    p=client.post('/api/projects').json();pid=p['project_id'];h={'Authorization':'Bearer '+p['access_token']}
    with clip.open('rb') as f: assert client.post(f'/api/projects/{pid}/upload',headers=h,files={'file':('clip.mp4',f)}).status_code==200
    url=f'/api/projects/{pid}/export'; clips=[{'start':1,'end':2},{'start':0,'end':1},{'start':1,'end':2}]
    assert client.post(url,headers=h,json={'format':'mp4','clips':[]}).status_code==422
    assert client.post(url,headers=h,json={'format':'mp4','clips':[{'start':0,'end':4}]}).status_code==422
    assert client.post(url,headers=h,json={'format':'mp4','clips':clips,'trim':{'start':0,'end':1}}).status_code==422
    for fmt in ['mp4','wav']:
        r=client.post(url,headers=h,json={'format':fmt,'include_video':fmt=='mp4','clips':clips}); assert r.status_code==202,r.text
        job=wait(client,'/api/exports/'+r.json()['export_id'],h);assert job['status']=='completed',job
        out=tmp_path/('sequence-out.'+fmt);out.write_bytes(client.get(job['download_url'],headers=h).content)
        samples=np.frombuffer(subprocess.check_output(['ffmpeg','-v','error','-i',str(out),'-ac','1','-ar','48000','-f','f32le','-']),dtype='<f4')
        for index,expected in enumerate((880,440,880)):
            chunk=samples[int((index+.2)*48000):int((index+.8)*48000)]
            freq=np.fft.rfftfreq(len(chunk),1/48000)[np.argmax(np.abs(np.fft.rfft(chunk)))]
            assert abs(freq-expected)<5
        if fmt=='mp4':
            info=json.loads(subprocess.check_output(['ffprobe','-v','error','-show_streams','-of','json',str(out)]))
            for stream in info['streams']:
                assert abs(float(stream['duration'])-3)<.12
            def frame(t):
                return subprocess.check_output(['ffmpeg','-v','error','-ss',str(t),'-i',str(out),'-frames:v','1','-f','rawvideo','-pix_fmt','rgb24','-'])
            a,b,c=(np.frombuffer(frame(t),dtype='u1').astype(float) for t in (.4,1.4,2.4))
            assert np.mean(abs(a-c)) < np.mean(abs(a-b))
    # Cutting the middle second joins the first and last second in both streams.
    r=client.post(url,headers=h,json={'format':'mp4','include_video':True,'clips':[{'start':0,'end':1},{'start':2,'end':3}]})
    assert r.status_code==202
    job=wait(client,'/api/exports/'+r.json()['export_id'],h);assert job['status']=='completed',job
    cut=tmp_path/'cut.mp4';cut.write_bytes(client.get(job['download_url'],headers=h).content)
    info=json.loads(subprocess.check_output(['ffprobe','-v','error','-show_streams','-of','json',str(cut)]))
    for stream in info['streams']: assert abs(float(stream['duration'])-2)<.12
    samples=np.frombuffer(subprocess.check_output(['ffmpeg','-v','error','-i',str(cut),'-ac','1','-ar','48000','-f','f32le','-']),dtype='<f4')
    for index,expected in enumerate((440,1320)):
        chunk=samples[int((index+.2)*48000):int((index+.8)*48000)]
        freq=np.fft.rfftfreq(len(chunk),1/48000)[np.argmax(np.abs(np.fft.rfft(chunk)))]
        assert abs(freq-expected)<5


def test_range_automation_is_audible_in_export_and_can_be_undone(client,tmp_path):
    import numpy as np
    clip=tmp_path/'tone.wav'
    subprocess.run(['ffmpeg','-v','error','-y','-f','lavfi','-i','sine=frequency=440:duration=3',str(clip)],check=True)
    p=client.post('/api/projects').json();pid=p['project_id'];h={'Authorization':'Bearer '+p['access_token']}
    with clip.open('rb') as f: assert client.post(f'/api/projects/{pid}/upload',headers=h,files={'file':('tone.wav',f)}).status_code==200
    jobid=client.post(f'/api/projects/{pid}/process',headers=h,json={'prompts':['tone']}).json()['job_id']
    job=wait(client,'/api/jobs/'+jobid,h); assert job['status']=='completed'
    stem=job['stems'][0]['id']
    automation=[{'time':.5,'value':1},{'time':.55,'value':0},{'time':1.45,'value':0},{'time':1.5,'value':1}]
    assert client.patch(f'/api/projects/{pid}/stems/{stem}',headers=h,json={'automation':automation}).status_code==200
    def render(name):
        response=client.post(f'/api/projects/{pid}/export',headers=h,json={'format':'wav','include_video':False}); assert response.status_code==202
        result=wait(client,'/api/exports/'+response.json()['export_id'],h);assert result['status']=='completed',result
        path=tmp_path/name;path.write_bytes(client.get(result['download_url'],headers=h).content)
        return np.frombuffer(subprocess.check_output(['ffmpeg','-v','error','-i',str(path),'-ac','1','-ar','48000','-f','f32le','-']),dtype='<f4')
    edited=render('edited.wav')
    def rms(data,start,end): return float(np.sqrt(np.mean(data[int(start*48000):int(end*48000)]**2)))
    assert rms(edited,.7,1.3) < rms(edited,1.8,2.4)*.01
    assert client.patch(f'/api/projects/{pid}/stems/{stem}',headers=h,json={'automation':[]}).status_code==200
    restored=render('restored.wav')
    assert rms(restored,.7,1.3) > rms(edited,.7,1.3)*10 + .001
