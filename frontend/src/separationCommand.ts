import type { EditScope } from './api/backend'
/** Separate auditory description from the requested edit, including timed mute. */
export function separationCommand(text: string): {description: string; remove: boolean; editScope?:EditScope} | null {
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
