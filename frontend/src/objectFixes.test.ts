import { describe, expect, it } from 'vitest'
import type { VisualMetadata, VisualObject } from './api/backend'
import { mergeTracks, objectTracks, removeTrack, renameTrack } from './objects'
import { onScreenMoments } from './moonpull/onscreen'

const obj = (track: number, label = 'person'): VisualObject => ({ detection_id: `d${track}-${Math.random()}`, track_id: track, label, bbox_normalized: [0, 0, 0.5, 0.5], mask: 'm' })
// Track 6 on screen 0–1 s, track 9 at 3–4 s, track 8 (a shadow) throughout; 10 fps.
const meta: VisualMetadata = {
  schema_version: '1', video_id: 'v', analysis_id: 'a', source_sha256: 's', source: 'x', width: 10, height: 10, fps: 10,
  processed_frames: 50, processed_duration_seconds: 5, discovery_mode: 'automatic', original_source_sha256: 'keep-me',
  frames: Array.from({ length: 50 }, (_, i) => ({
    frame_index: i, timestamp_seconds: i / 10, valid_until_seconds: (i + 1) / 10,
    objects: [obj(8), ...(i < 10 ? [obj(6)] : []), ...(i >= 30 && i < 40 ? [obj(9)] : []), ...(i === 5 ? [obj(9)] : [])],
  })),
} as VisualMetadata

describe('fixing detections', () => {
  it('removes a track everywhere and keeps the rest of the document', () => {
    const next = removeTrack(meta, 8)
    expect(objectTracks(next).map(t => t.trackId)).toEqual([6, 9])
    expect((next as unknown as { original_source_sha256: string }).original_source_sha256).toBe('keep-me')
    expect(objectTracks(meta).length).toBe(3) // the original is untouched
  })

  it('merges two tracks into one entry whose on-screen time is both', () => {
    const next = mergeTracks(meta, 6, 9)
    const tracks = objectTracks(next)
    expect(tracks.map(t => t.trackId)).toEqual([6, 8])
    const six = tracks.find(t => t.trackId === 6)!
    expect(six.firstSeconds).toBe(0); expect(six.lastSeconds).toBeCloseTo(3.9)
    // A frame where both were seen keeps one observation of the merged track.
    expect(next.frames[5].objects.filter(o => o.track_id === 6)).toHaveLength(1)
    const moments = onScreenMoments(next, [6], 5)
    expect(moments.map(m => [m.start, m.end])).toEqual([[0, 1], [3, 4]])
  })

  it('names a track without changing its detector label, and the name survives a merge', () => {
    const named = renameTrack(meta, 9, 'Allie')
    expect(objectTracks(named).find(t => t.trackId === 9)).toMatchObject({ name: 'Allie', label: 'person' })
    const merged = mergeTracks(named, 6, 9)
    expect(objectTracks(merged).find(t => t.trackId === 6)?.name).toBe('Allie')
    expect(onScreenMoments(merged, [6], 5)[0].note).toBe('Allie on screen')
    expect(objectTracks(renameTrack(named, 9, '  ')).find(t => t.trackId === 9)?.name).toBeUndefined()
  })
})
