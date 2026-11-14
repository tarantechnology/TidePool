import {describe,expect,it} from 'vitest'
import {replacementFollowup} from './replacementFollowup'
describe('replacement time follow-up',()=>{
  it('uses the supplied time with the earlier request',()=>expect(replacementFollowup('Replace every punch with two quacks.','at 4.5 seconds')).toBe('Replace every punch with two quacks at 4.5 seconds'))
  it('does not reinterpret a new instruction as a time',()=>expect(replacementFollowup('replace the dog with a quack','make it louder')).toBeNull())
})
