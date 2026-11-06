import re

def parse(text):
    text = text.strip().rstrip('.!?')
    if re.fullmatch(r'export (?:the )?video',text,re.I): return {'action':'export'}
    match = re.fullmatch(r'(unmute|mute) (?:the )?(.+?)(?: from (\d+(?:\.\d+)?) to (\d+(?:\.\d+)?) seconds?)?',text,re.I)
    if match:
        action,name,start,end=match.groups(); action=action.lower()
        if start:
            if action != 'mute' or float(start)>=float(end): return None
            return {'action':'mute_range','stem_name':name,'start':float(start),'end':float(end)}
        return {'action':action,'stem_name':name}
    match=re.fullmatch(r'(?:lower|set) (?:the )?(.+?) volume to (\d+(?:\.\d+)?)%',text,re.I)
    if match:
        name,value=match.groups(); value=float(value)/100
        return {'action':'set_volume','stem_name':name,'volume':value} if value<=2 else None
    match=re.fullmatch(r'make (?:the )?(.+?) louder',text,re.I)
    if match: return {'action':'louder','stem_name':match.group(1)}
    return None
