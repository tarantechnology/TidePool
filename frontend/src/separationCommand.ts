import type { EditScope } from './api/backend'
/** Separate auditory description from the requested edit, including timed mute. */
export function separationCommand(text: string): {description: string; remove: boolean; editScope?:EditScope; processingTime?:number} | null {
  const point = /^\s*(remove|mute|isolate|separate|lower|reduce)\s+(?:the\s+)?(.+?)\s+(?:at|around|near)\s+(\d+(?:\.\d+)?)\s*(?:seconds?|secs?|s)?\s*[.!]?\s*$/i.exec(text)
  if (point) {
    const action=point[1].toLowerCase(), sound=point[2].trim()
    if (!sound || /\b(?:and|but|except|all|throughout|entire)\b/i.test(sound)) return null
    const remove=action==='remove'||action==='mute'
    return {description:sound.toLowerCase(), remove, processingTime:Number(point[3]), ...(action==='lower'||action==='reduce'?{editScope:{action:'lower' as const}}:{})}
  }
  const match = /^\s*(remove|mute|isolate|separate)\s+(?:the\s+)?(.+?)\s*[.!]?\s*$/i.exec(text)
  if (!match) return null
  let description = match[2].replace(/[.!]+$/, '').trim()
  const timed = /^(.*?)\s+from\s+(\d+(?:\.\d+)?)\s+to\s+(\d+(?:\.\d+)?)\s+seconds?$/i.exec(description)
  const remove=['remove','mute'].includes(match[1].toLowerCase())
  if(timed){
    const start=Number(timed[2]),end=Number(timed[3]);description=timed[1].trim()
    if(!remove || end<=start || !description)return null
    return {description:description.toLowerCase(),remove,editScope:{action:'mute',interval:{start,end}}}
  }
  if (!description || /\b(?:from|between|until|after|before|when|while|then|and|at)\b|\d/i.test(description)) return null
  return {description:description.toLowerCase(), remove}
}
