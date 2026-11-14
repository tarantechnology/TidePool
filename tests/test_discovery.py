from app.discovery import group_frames


def test_grouping_deduplicates_synonyms_and_keeps_separate_occurrences():
    frames=[(0, .96, {'Bow-wow':.7, 'Bark':.8, 'Speech':.6}), (.48,1.44,{'Bark':.9}), (4,4.96,{'Bark':.8}), (5,5.96,{'Music':.1})]
    sources={s['label']:s for s in group_frames(frames, 5)}
    assert sources['dog barking']['intervals'] == [{'start':0,'end':1.44},{'start':4,'end':4.96}]
    assert sources['person speaking']['intervals'] == [{'start':0,'end':.96}]
    assert 'music' not in sources


def test_localize_uses_only_a_small_neighborhood_and_nearby_match(monkeypatch):
    import app.discovery as discovery
    calls=[]
    def discover(path,duration,start,end):
        calls.append((start,end))
        return {'sources':[{'label':'chopping','intervals':[{'start':3.5,'end':4.5},{'start':7,'end':8}]}]}
    monkeypatch.setattr(discovery,'discover',discover)
    result=discovery.locate_near('audio.wav',30,'knife chopping',4)
    assert calls==[(0.0,9.0)]
    assert result['method']=='yamnet' and result['interval']=={'start':3.5,'end':4.5}


def test_localize_low_evidence_or_unknown_uses_explicit_fallback(monkeypatch):
    import app.discovery as discovery
    monkeypatch.setattr(discovery,'discover',lambda *args:{'sources':[{'label':'chopping','intervals':[{'start':8,'end':9}]}]})
    result=discovery.locate_near('audio.wav',30,'knife chopping',4)
    assert result['method']=='fallback' and result['interval']=={'start':3,'end':5}
    assert 'no matching' in result['notice'].lower()
    other=discovery.locate_near('audio.wav',30,'an unsupported sound',0)
    assert other['interval']=={'start':0,'end':1} and other['method']=='fallback'


def test_onnx_unavailable_and_unknown_sound_have_honest_fallback(monkeypatch):
    import app.discovery as discovery
    monkeypatch.delenv('SOUNDCANVAS_YAMNET_ONNX',raising=False)
    monkeypatch.delenv('SOUNDCANVAS_YAMNET_MODEL',raising=False)
    result=discovery.locate_near('missing.wav',12,'dog barking',4)
    assert result['method']=='fallback' and result['interval']=={'start':3,'end':5}
    assert 'unavailable' in result['notice']
