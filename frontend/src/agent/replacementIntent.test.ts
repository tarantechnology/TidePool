import {describe,expect,it} from 'vitest'
import {requestsSoundReplacement} from './replacementIntent'
describe('replacement intent routing',()=>{
  it.each(['replace all knife chopping sounds with quack sounds','I want you to replace all of the knife chopping sounds with quacks','Could you swap the barks for a quack?','make every punch sound like a quack','add a quack at 4 seconds','Add a gong sound after the word "quant" has been said','Add a gong every time we see person 5','Every time person 5 appears, play a gong sound.','play a gong whenever person #5 shows up'])('routes %s to replacement',text=>expect(requestsSoundReplacement(text)).toBe(true))
  it.each(['separate the knife chopping sounds','remove the dog at 4 seconds','make the music louder'])('does not reroute %s',text=>expect(requestsSoundReplacement(text)).toBe(false))
})
