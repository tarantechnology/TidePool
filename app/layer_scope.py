"""Resolve an explicitly named project layer without guessing among matches."""
import re

_SCOPE=re.compile(r'\b(?:in|on|from)\s+(?:the\s+)?([\w\'’ -]{1,100}?)\s+(?:audio\s+)?(?:layer|track)\b',re.I)


def extract_layer_scope(text):
    matches=list(_SCOPE.finditer(text))
    if len(matches)>1:raise ValueError('Please specify one source layer at a time.')
    if not matches:return text,None
    match=matches[0];name=match.group(1).strip()
    if re.search(r'\b(?:seconds?|secs?|from|between|before|after)\b',name,re.I):raise ValueError('Please name the source layer separately from the time selection.')
    cleaned=' '.join((text[:match.start()]+' '+text[match.end():]).split())
    return cleaned,name


def resolve_layer(name,rows):
    key=' '.join(name.casefold().split())
    matches=[r for r in rows if key in {' '.join((r.get('name') or '').casefold().split()),' '.join((r.get('label') or '').casefold().split())}]
    if not matches:raise ValueError(f'No layer named “{name}” was found. Choose an existing layer.')
    if len(matches)>1:raise ValueError(f'Several layers are named “{name}”. Choose one source layer explicitly.')
    return matches[0]
