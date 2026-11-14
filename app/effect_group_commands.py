"""Resolve bounded group mixer requests to existing project stem IDs."""
import re
from .effect_followup import matching_effects, resolve_effect_reference


def plan_group_command(text, rows):
    if not any(r.get('effect_group_id') for r in rows if r.get('role')=='effect'):return None
    text=text.strip().rstrip('.!?')
    moving=re.fullmatch(r'move\s+(?:the\s+)?(first|second|third|fourth|fifth|\d+(?:st|nd|rd|th))\s+(.+?)\s+(earlier|later)(?:\s+by\s+(\d+(?:\.\d+)?)\s*(seconds?|secs?|s|milliseconds?|ms))?',text,re.I)
    if moving:
        ordinal,reference,direction,amount,unit=moving.groups()
        if amount is None:return {'kind':'clarify','description':None,'remove':False,'actions':[],'message':'How many seconds should I move it?'}
        matches=matching_effects(rows,reference)
        if not matches:
            try:matches=resolve_effect_reference(reference,rows)
            except ValueError as exc:return {'kind':'clarify','description':None,'remove':False,'actions':[],'message':str(exc)}
        if not matches:return None
        groups={r.get('effect_group_id') for r in matches}
        if len(groups)!=1 or None in groups:return {'kind':'clarify','description':None,'remove':False,'actions':[],'message':'Several sound groups match. Please choose one.'}
        indices={'first':0,'second':1,'third':2,'fourth':3,'fifth':4}
        position=indices.get(ordinal.lower())
        if position is None:position=int(re.match(r'\d+',ordinal).group())-1
        ordered=sorted(matches,key=lambda r:(r['start_offset'],r['id']))
        if position<0 or position>=len(ordered):return {'kind':'clarify','description':None,'remove':False,'actions':[],'message':'That group does not have that many clips.'}
        row=ordered[position];delta=float(amount)/(1000 if unit.lower().startswith(('milli','ms')) else 1)
        offset=row['start_offset']+delta*(1 if direction.lower()=='later' else -1)
        if offset<0:return {'kind':'clarify','description':None,'remove':False,'actions':[],'message':'That would move the clip before the beginning.'}
        return {'kind':'clip_edit','description':None,'remove':False,'actions':[],'clip_edit':{'stem_id':row['id'],'operation':'move','start_offset':offset},'message':'Moving the selected clip.'}
    command=re.fullmatch(r'(mute|unmute|solo|unsolo)\s+(?:(?:all|the)\s+)*(.*)',text,re.I)
    loud=re.fullmatch(r'make\s+(?:(?:all|the)\s+)*(.*?)\s+(louder|quieter)',text,re.I)
    if command: operation=command.group(1).lower();reference=command.group(2).strip()
    elif loud: operation=loud.group(2).lower();reference=loud.group(1).strip()
    else:return None
    if not reference:return None
    matches=matching_effects(rows,reference)
    if not matches:
        try:matches=resolve_effect_reference(reference,rows)
        except ValueError as exc:return {'kind':'clarify','description':None,'remove':False,'actions':[],'message':str(exc)}
    if not matches:return None
    groups={r.get('effect_group_id') for r in matches}
    if len(groups)!=1 or None in groups:
        return {'kind':'clarify','description':None,'remove':False,'actions':[],'message':'Several sound groups match. Please choose the group by name.'}
    actions=[]
    for row in matches:
        if operation in ('mute','unmute'):actions.append({'type':'set_mute','stemId':row['id'],'muted':operation=='mute'})
        elif operation in ('solo','unsolo'):actions.append({'type':'set_solo','stemId':row['id'],'solo':operation=='solo'})
        else:actions.append({'type':'set_gain','stemId':row['id'],'value':min(2,max(0,row['volume']*(1.25 if operation=='louder' else .8)))})
    return {'kind':'edit','description':None,'remove':False,'actions':actions,'message':f'Updated {len(actions)} clips in the sound group.'}
