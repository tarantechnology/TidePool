import { describe, expect, it } from 'vitest'
import { groupActions, timelineRows } from './effectGroups'
import type { Stem, TrackState } from '../types'

const stem=(id:string,group?:string):Stem=>({id,label:id,kind:'sfx',url:'x',duration:1,effectGroupId:group,effectLabel:'gong'})
const track=(stemId:string,gain:number,muted=false):TrackState=>({stemId,gain,muted,solo:false,automation:[]})
describe('effect groups',()=>{
  it('groups only matching identities and expands independent clips',()=>{
    const stems=[stem('one','group'),stem('other'),stem('two','group'),stem('same-name','different')]
    const rows=timelineRows(stems,new Set())
    expect(rows.map(r=>r.type)).toEqual(['group','stem','stem'])
    expect(rows[0].type==='group' && rows[0].stems.map(s=>s.id)).toEqual(['one','two'])
    expect(timelineRows(stems,new Set(['group'])).map(r=>r.type)).toEqual(['group','stem','stem','stem','stem'])
  })
  it('scales relative levels, and limits actions to selected group members',()=>{
    const tracks=[track('one',1),track('two',.5,true),track('other',.8)]
    expect(groupActions(tracks,['one','two'],{type:'gain',value:1.5})).toEqual([{type:'set_gain',stemId:'one',value:1.5},{type:'set_gain',stemId:'two',value:.75}])
    expect(groupActions(tracks,['one','two'],{type:'solo',value:true})).toEqual([{type:'set_solo',stemId:'one',solo:true},{type:'set_solo',stemId:'two',solo:true}])
  })
})
