import {describe,expect,it} from 'vitest'
import {requestsSoundReplacement} from './replacementIntent'
describe('replacement intent routing',()=>{
  it.each(['replace all knife chopping sounds with quack sounds','I want you to replace all of the knife chopping sounds with quacks','Could you swap the barks for a quack?','make every punch sound like a quack','add a quack at 4 seconds','Add a gong sound after the word "quant" has been said'])('routes %s to replacement',text=>expect(requestsSoundReplacement(text)).toBe(true))
  it.each(['separate the knife chopping sounds','remove the dog at 4 seconds','make the music louder'])('does not reroute %s',text=>expect(requestsSoundReplacement(text)).toBe(false))
})
