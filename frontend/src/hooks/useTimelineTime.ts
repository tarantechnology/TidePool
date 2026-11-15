import {useEffect,useState} from 'react'
import type {AudioEngine} from '../audio/engine'
export function useTimelineTime(engine:AudioEngine|null){
  const [time,setTime]=useState(0)
  useEffect(()=>{if(!engine)return;let frame=0;const tick=()=>{setTime(engine.getTimelineTime());frame=requestAnimationFrame(tick)};tick();return()=>cancelAnimationFrame(frame)},[engine])
  return time
}
