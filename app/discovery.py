"""Optional YAMNet event suggestions. Classification is not separation."""
import csv
import os
import subprocess
from functools import lru_cache

# Deliberately conservative mappings: related classes describe one logical source.
LABELS = {
    'Speech': 'person speaking', 'Conversation': 'person speaking',
    'Music': 'music', 'Bark': 'dog barking', 'Bow-wow': 'dog barking',
    'Traffic noise, roadway noise': 'traffic noise', 'Footsteps': 'footsteps',
    'Siren': 'siren', 'Bird vocalization, bird call, bird song': 'birds calling',
    'Rain': 'rain', 'Wind': 'wind', 'Water': 'water', 'Engine': 'engine',
    'Laughter': 'laughter', 'Applause': 'applause', 'Gunshot, gunfire': 'gunshots',
    'Chop': 'chopping', 'Sizzle': 'sizzling', 'Walk, footsteps': 'footsteps',
    'Sneeze': 'sneezing', 'Cough': 'coughing', 'Dog': 'dog',
    'Boing': 'boing', 'Fart': 'fart', 'Splash, splatter': 'splash',
    'Whoosh, swoosh, swish': 'whoosh', 'Bang': 'bang', 'Sound effect': 'sound effect',
    'Giggle': 'laughter', 'Chatter': 'person speaking',
}

@lru_cache(maxsize=1)
def load_model(path):
    try:
        import tensorflow as tf
    except ImportError as exc:
        raise RuntimeError('Sound discovery needs TensorFlow and a local YAMNet SavedModel; see INTEGRATION.md.') from exc
    try: tf.config.set_visible_devices([], 'GPU')
    except RuntimeError: pass  # TensorFlow may already be initialized.
    with tf.device('/CPU:0'):
        model = tf.saved_model.load(path)
    with open(model.class_map_path().numpy().decode(), newline='') as f:
        names = [r['display_name'] for r in csv.DictReader(f)]
    return model, names


def group_frames(frames, duration, threshold=0.3):
    """Merge adjacent classifier windows, never implying precise event boundaries."""
    grouped = {}
    for start, end, scores in frames:
        best = {}
        for name, score in scores.items():
            label = LABELS.get(name)
            if label and score >= threshold:
                best[label] = max(best.get(label, 0), float(score))
        for label, score in best.items():
            entries = grouped.setdefault(label, [])
            end = min(end, duration)
            if end <= start: continue
            if entries and start <= entries[-1][1] + 0.05:
                entries[-1][1] = max(entries[-1][1], end)
                entries[-1][2] = max(entries[-1][2], score)
            else: entries.append([start, end, score])
    return [{'label':label, 'prompt':label, 'intervals':[{'start':a,'end':b} for a,b,_ in entries],
             'interval_scores':[e[2] for e in entries], 'score':max(e[2] for e in entries)} for label, entries in grouped.items()]


@lru_cache(maxsize=1)
def load_onnx(path, class_map):
    try:
        import onnxruntime as ort
    except ImportError as exc:
        raise RuntimeError('Sound discovery needs onnxruntime for the configured YAMNet model.') from exc
    model = ort.InferenceSession(path, providers=['CPUExecutionProvider'])
    with open(class_map, newline='') as f:
        names = [r['display_name'] for r in csv.DictReader(f)]
    return model, names


def discover(path, duration, start=0.0, end=None):
    onnx_path = os.getenv('SOUNDCANVAS_YAMNET_ONNX', '').strip()
    model_path = os.getenv('SOUNDCANVAS_YAMNET_MODEL', '').strip()
    if not onnx_path and not model_path: raise RuntimeError('Sound discovery is not configured. Set SOUNDCANVAS_YAMNET_ONNX or SOUNDCANVAS_YAMNET_MODEL.')
    import numpy as np
    if onnx_path:
        class_map=os.getenv('SOUNDCANVAS_YAMNET_CLASSES', '').strip()
        if not class_map: raise RuntimeError('Set SOUNDCANVAS_YAMNET_CLASSES to the YAMNet class map.')
        model, names = load_onnx(onnx_path, class_map)
    else:
        model, names = load_model(model_path)
        import tensorflow as tf
    frames=[]
    # Bounded memory: process 30-second sections with a small overlap.
    end = min(duration, end if end is not None else duration)
    offset=max(0.0,start)
    while offset < end:
        length=min(30.0, end-offset)
        cmd=['ffmpeg','-v','error','-ss',str(offset),'-t',str(length),'-i',str(path),'-vn','-ac','1','-ar','16000','-f','f32le','pipe:1']
        result=subprocess.run(cmd, capture_output=True, check=True)
        samples=np.frombuffer(result.stdout,dtype='<f4').copy()
        if samples.size == 0: break
        if onnx_path:
            scores=model.run(None,{model.get_inputs()[0].name:samples})[0]
        else:
            with tf.device('/CPU:0'):
                scores, _, _ = model(samples)
            scores=scores.numpy()
        for i, row in enumerate(scores):
            start=offset+i*0.48
            frames.append((start,min(start+0.96,end),{names[j]:float(row[j]) for j in range(min(len(names),len(row))) if names[j] in LABELS}))
        if offset+length >= end: break
        offset += 29.28
    return {'sources':group_frames(frames,duration), 'model':'yamnet', 'is_mock':False,
            'notice':'Suggestions are approximate classifier windows, not separated layers or calibrated confidence.'}


# Deliberately explicit mappings: do not pretend a generic event detector understands
# arbitrary source identity or a visually observed object.
QUERY_LABELS = {
    'bark':'dog barking', 'dog':'dog barking', 'chop':'chopping', 'knife':'chopping',
    'sizzl':'sizzling', 'fry':'sizzling', 'speech':'person speaking', 'speak':'person speaking',
    'talk':'person speaking', 'narrat':'person speaking', 'voice':'person speaking',
    'music':'music', 'footstep':'footsteps', 'siren':'siren', 'bird':'birds calling',
    'rain':'rain', 'traffic':'traffic noise', 'laugh':'laughter', 'sneez':'sneezing',
    'cough':'coughing', 'wind':'wind', 'water':'water', 'engine':'engine',
}

def locate_near(path, duration, query, point):
    """Propose a local event range, or transparently fall back to +/- one second."""
    if duration <= 0 or point < 0 or point > duration: raise ValueError('Timestamp is outside the audio timeline.')
    fallback={'start':max(0.0,point-1.0),'end':min(duration,point+1.0)}
    if fallback['end'] <= fallback['start']: fallback={'start':max(0.0,duration-2.0),'end':duration}
    query=query.lower()
    label=next((v for k,v in QUERY_LABELS.items() if k in query),None)
    if not label:
        return {'interval':fallback,'method':'fallback','notice':'No matching detector category; using a two-second range around the requested time.'}
    left=max(0.0,point-5.0);right=min(duration,point+5.0)
    try:
        found=discover(path,duration,left,right)
    except Exception:  # Optional detector errors must not prevent bounded separation.
        return {'interval':fallback,'method':'fallback','notice':'Sound detection is unavailable; using a two-second range around the requested time.'}
    ranges=[i for item in found['sources'] if item['label']==label for i in item['intervals']]
    def distance(i): return max(i['start']-point,0.0,point-i['end'])
    nearby=sorted(ranges,key=distance)
    if not nearby or distance(nearby[0])>0.5:
        return {'interval':fallback,'method':'fallback','notice':'No matching event was detected near this time; using a two-second range. Adjust it if needed.'}
    selected=nearby[0]
    return {'interval':{'start':max(left,selected['start']),'end':min(right,selected['end'])},'method':'yamnet',
            'notice':'The detector suggests an approximate range; check it before applying.'}
