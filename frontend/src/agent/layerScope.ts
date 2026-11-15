import type { SavedStem } from '../api/backend'

/** Only an explicitly named layer/track changes the source; never guess a layer. */
export function extractLayerScope(text:string):{text:string;name:string}|null {
  const pattern=/\b(?:in|on|from)\s+(?:the\s+)?([\w'’ -]{1,100}?)\s+(?:audio\s+)?(?:layer|track)\b/ig
  const matches=[...text.matchAll(pattern)]
  if (!matches.length) return null
  if (matches.length!==1 || /\b(?:seconds?|secs?|from|between|before|after)\b/i.test(matches[0][1])) throw new Error('Please specify one source layer separately from the time selection.')
  const match=matches[0],index=match.index ?? 0
  return {name:match[1].trim(),text:(text.slice(0,index)+' '+text.slice(index+match[0].length)).replace(/\s+/g,' ').trim()}
}

export function resolveSourceLayer(name:string,rows:SavedStem[]):SavedStem {
  const key=name.toLowerCase().replace(/\s+/g,' ').trim()
  const matches=rows.filter(r=>[r.name,r.label || ''].some(value=>value.toLowerCase().replace(/\s+/g,' ').trim()===key))
  if (!matches.length) throw new Error(`No layer named “${name}” was found. Choose an existing layer.`)
  if (matches.length>1) throw new Error(`Several layers are named “${name}”. Choose one source layer explicitly.`)
  return matches[0]
}
