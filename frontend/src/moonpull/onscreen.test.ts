import { describe, expect, it } from 'vitest'
import type { VisualMetadata } from '../api/backend'
import { onScreenMoments, parseOnScreenCommand } from './onscreen'

function metadata(seen: Record<number, number[]>, frames = 300, fps = 10): VisualMetadata {
  return {
    schema_version: '1.0', video_id: 'v', analysis_id: 'a', source_sha256: 's', source: 'x', width: 10, height: 10, fps,
    processed_frames: frames, processed_duration_seconds: frames / fps,
    frames: Array.from({ length: frames }, (_, i) => ({
      frame_index: i, timestamp_seconds: i / fps, valid_until_seconds: (i + 1) / fps,
      objects: Object.entries(seen).filter(([, idx]) => idx.includes(i)).map(([track]) => ({ detection_id: `${track}-${i}`, track_id: Number(track), label: 'person', bbox_normalized: [0, 0, 1, 1], mask: {} })),
    })),
  }
}
const range = (a: number, b: number) => Array.from({ length: b - a }, (_, k) => a + k)

describe('onScreenMoments', () => {
  it('turns runs of frames into moments, bridging short gaps and dropping flashes', () => {
    const m = metadata({ 3: [...range(10, 40), ...range(43, 60), ...range(100, 102), ...range(200, 250)], 4: range(0, 300) })
    const moments = onScreenMoments(m, [3], 30)
    expect(moments.map((x) => [x.start, x.end])).toEqual([[1, 6], [20, 25]])
    expect(moments[0].note).toBe('person #3 on screen')
  })
  it('joins several tracks of the same person (a cut gives them a new number)', () => {
    const m = metadata({ 3: range(10, 30), 7: range(50, 70) })
    expect(onScreenMoments(m, [3, 7], 30).map((x) => [x.start, x.end])).toEqual([[1, 3], [5, 7]])
    expect(onScreenMoments(m, [], 30)).toEqual([])
  })
})

describe('parseOnScreenCommand', () => {
  it('reads the usual phrasings', () => {
    expect(parseOnScreenCommand('whenever the tall guy is on screen, add funny music')).toEqual({ subject: 'the tall guy', sound: 'funny music' })
    expect(parseOnScreenCommand('play a heroic theme every time Batman appears')).toEqual({ subject: 'Batman', sound: 'heroic theme' })
    expect(parseOnScreenCommand('when the dog is in the frame play a squeaky toy sound.')).toEqual({ subject: 'the dog', sound: 'squeaky toy sound' })
    expect(parseOnScreenCommand('whenever she shows up')).toEqual({ subject: 'she', sound: '' })
  })
  it('leaves other commands alone', () => {
    expect(parseOnScreenCommand('mute the rain from 3 to 5')).toBeNull()
    expect(parseOnScreenCommand('when he speaks, lower the music')).toBeNull()
    expect(parseOnScreenCommand('make it cinematic')).toBeNull()
    expect(parseOnScreenCommand('Every time person 5 appears, play a gong sound.')).toBeNull()
  })
})
