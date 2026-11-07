# Contracts

What the SoundCanvas editor (Person 3) expects from, and offers to, the other workstreams.

Each contract below has one of three labels:

- **SHARED**: defined in this repo, in `src/types.ts`.
- **DOCUMENTED**: read from upstream `INTEGRATION.md` and `app/main.py` at AkhilNam/SoundCanvas commit 9fc2216. This client has only run against mocks of it, not a live server.
- **ASSUMED**: invented here because nothing documents it yet. Needs confirmation.

## Shared types (SHARED)

Defined in `src/types.ts`.

```ts
type StemKind = 'dialogue' | 'sfx' | 'music' | 'ambience'

interface Stem { id: string; label: string; kind: StemKind; url: string; duration: number }
interface Keyframe { t: number; value: number }
interface TrackState { stemId: string; gain: number; muted: boolean; solo: boolean; automation: Keyframe[] }
interface EditList { version: 1; tracks: TrackState[] }

type EditAction =
  | { type: 'set_gain'; stemId: string; value: number }
  | { type: 'set_mute'; stemId: string; muted: boolean }
  | { type: 'set_solo'; stemId: string; solo: boolean }
  | { type: 'set_automation'; stemId: string; keyframes: Keyframe[] }
  | { type: 'clear_automation'; stemId: string }
  | { type: 'follow_intensity'; stemId: string; sourceStemIds: string[]; minValue?: number; maxValue?: number }
```

Gains are multipliers from 0 to 2. Automation is piecewise linear in timeline seconds and multiplies the fader gain.

### Additions made in this sprint

All additive. No existing field or union member changed.

- `EditAction` gained `follow_intensity`. It makes a stem's gain follow how loud the source stems are. That is a loudness proxy, not emotion detection. It is resolved to `set_automation` in the browser (`src/audio/resolve.ts`) because that needs decoded audio, so it never reaches an edit list or an exporter. `applyAction` treats an unresolved one as a no-op. `minValue` is the gain at low loudness and `maxValue` at high loudness. `maxValue` below `minValue` ducks.

## Edit list

`serializeEditList` writes `EditList` as JSON and `parseEditList` validates it, sorting keyframes. The **Export edit list (JSON)** button downloads the same JSON.

The upstream backend does not take an edit list. It stores the same data per stem through PATCH, so `src/api/backend.ts` maps one to the other:

| TrackState | Backend stem |
|---|---|
| `gain` | `volume` |
| `muted`, `solo` | `muted`, `solo` |
| `automation[].t` | `automation[].time` |
| `automation[].value` | `automation[].value` |
| `stemId` | stem `id` in the path |

The backend stores automation and app/media.py renders it the way `envelopeAt` does: linear segments, holding both ends, multiplied by volume. So a curve maps across exactly, with one exception.

### What does not reach the backend

The backend requires strictly increasing keyframe times, so it cannot hold an exact step: two keyframes at the same time. The editor produces these. "Mute Speaker B at 12 seconds" ends the stem with `{20,0},{20,1}`, a mute from 0 starts with `{0,1},{0,0}`, a mute that runs to the end of the stem ends with one, and the agent can send them.

Collapsing a step is not a safe fallback. It keeps only the later keyframe, so the segment before it becomes a ramp: that mute would fade back in over 8 seconds on export. So editor sync never collapses. `toSyncPatch` leaves `automation` out of the PATCH for that stem. Volume, mute and solo still go, the backend keeps whatever curve it last had for that stem, and the track header shows "not synced" while a job is loaded.

`trackToStemPatch` still collapses, because `server/backend-contract.test.ts` pins that behaviour through `patchStem`. Editor sync does not use either.

Also not sent, because the backend has no field for them: stem kinds. `start_offset` is left out of every PATCH, so the backend keeps its value.

## Agent request and response (ASSUMED between browser and agent server)

Both ends are in this repo, so this contract is internal for now.

`POST {VITE_AGENT_URL}` with:

```json
{
  "prompt": "make the siren louder",
  "stems": [{ "id": "siren", "label": "Siren", "kind": "sfx", "duration": 20 }],
  "tracks": [{ "stemId": "siren", "gain": 1, "muted": false, "solo": false, "automation": [] }],
  "duration": 20
}
```

Response:

```json
{ "actions": [{ "type": "set_gain", "stemId": "siren", "value": 1.5 }], "message": "Raised the siren." }
```

The browser treats the response as untrusted. `validateActions` allows at most 20 actions, known types and stems only, finite values in range, and 1 to 64 keyframes with `t` inside the timeline. Unknown fields are dropped. On a network failure, a timeout (15s) or invalid output, `HttpAgent` falls back to the regex parser and says so in the message. An empty `actions` list is valid and is how the agent asks a question.

### Agent ownership is unresolved

Who owns prompt-to-edit is not settled with the team: Person 3 (`src/agent` and `server/agent-server.mjs`) or Person 4 (the brief mentions LangChain). The upstream backend at commit 9fc2216 has no LangChain. It has a small regex command parser behind `POST /api/webhooks/local`, which edits stored stems rather than returning actions, so the two do not overlap today. This needs a team decision.

The browser only depends on the `Agent` interface in `src/agent/index.ts`, so the implementation can move without touching the editor.

## Backend API (DOCUMENTED)

Every call except project creation needs `Authorization: Bearer <project token>`, including audio downloads, so the client fetches audio as a blob. Errors are `{"detail": "message"}`, or a detail array for validation errors. The base URL comes from `VITE_API_BASE_URL`: `/api` with the Vite dev proxy, which forwards to `BACKEND_URL` with the path unchanged because the documented routes already start with `/api`, or the backend's own URL once it allows the editor's origin.

| Used by | Call | Notes |
|---|---|---|
| `getJob`, `HttpStemSource`, `JobStatusCard` | `GET /api/jobs/{id}` | `{ status, progress, error, stems }`. `status` is `queued`, `processing`, `completed` or `failed`. `progress` is 0 to 100. |
| `fetchStemBlobUrl` | `GET` the stem's `audio_url` | Path relative to the base URL. |
| `sendStemPatch` through `StemSync`, `patchStem` | `PATCH /api/projects/{id}/stems/{stem_id}` | `{ volume, muted, solo, automation }`. `automation` is omitted for a curve with a step. An omitted field keeps its stored value, and an empty list clears it. |

`project_id` on the job response is read from `get_job` in app/main.py, which returns the whole jobs row. INTEGRATION.md does not list it, so it needs confirming. Without it stems still load, edits stay in the browser, and the status chip says so.

### Edit sync

`StemSync` (`src/api/stemSync.ts`) runs only for stems loaded from a job. Mock and file stems have no sync target, so they make no calls. It follows `tracks`, so undo and redo sync too.

- Each stem is debounced for 500ms, longer than the 400ms undo coalescing window, so a slider drag is one write.
- Each write carries the stem's whole state, and a stem has one request in flight at most, so writes cannot land out of order and the last one wins.
- The state at load is the baseline, so loading sends nothing. Stored state is not loaded (see below), so the first edit to a stem replaces what the backend held for it.
- A failure never touches local tracks. The status chip shows the error, and the stem is sent again on the next change to any stem. After a failure the backend's state counts as unknown, because a timeout can land after the write applied, so the next state is always sent.
- Replacing the stems, or leaving the editor inside the app, sends what is still unsent once, and a sync for new stems waits for those writes to finish. If one fails, the chip says "Previous stems" until the next load.
- Closing or reloading the tab runs none of this, so edits from the last 500ms, and any stem waiting to retry, are lost with no warning.

`toJobStatus` maps the backend `status` to the UI's `JobStatus.state`, turning `completed` into `done`. An unknown status is an error, not a guess.

The sprint brief assumed `GET {base}/jobs/{jobId}/stems` and a `state` field. Neither exists upstream, so the client follows the documented API instead.

`server/backend-contract.test.ts` checks all of this against a running backend. It has not been run. It skips unless `BACKEND_URL` is set:

```sh
BACKEND_URL=http://localhost:8000 npx vitest run server/backend-contract
```

## Export (Person 4 owns this)

Person 4 owns the real export. `src/export/ffmpeg.ts` is a reference implementation to stop preview and export drifting apart. It is tested against the preview math and against real ffmpeg renders.

Finding for Person 4: `volume=...:eval=frame` holds the gain for a whole audio frame, about 43ms for decoded WAV here, so a steep curve rendered about 40ms late against the preview. `app/media.py` at upstream 9fc2216 uses the same filter, so it likely has the same lag (read from the code, not measured on that exporter). Inserting `asetnsamples=n=256` before `volume` fixed it in the reference.

The preview also applies a 0.7 master gain that neither exporter applies, so previews play quieter than exports.

## What the other workstreams need to confirm

### Person 4 (backend)

1. CORS. `app/main.py` has no CORS middleware, so browser calls with an Authorization header from `http://localhost:5173` should fail preflight (read from the code, not tested). In development the Vite proxy avoids this: set `VITE_API_BASE_URL=/api`. Without the proxy, and for any deployed build, Person 4 must add `CORSMiddleware` allowing the editor's origin.
2. Whether the `volume=...:eval=frame` lag above should be fixed in `app/media.py`.
3. `start_offset` and `mute_ranges` exist on backend stems. The editor has no equivalent, so it neither shows nor preserves their effect. `mute_ranges` set through the webhook will not appear in the preview.
4. Stored stem state is not loaded into the editor. It starts every job at unity gain with no automation, so a first edit overwrites a stem's stored volume, mute, solo and automation with the editor's.
5. The agent ownership question above.
6. Whether `project_id` on `GET /api/jobs/{id}` is part of the contract. Editor sync depends on it.
7. Exact steps in automation (see "What does not reach the backend"). Either the backend accepts equal times, which `app/media.py` would need to guard against because each segment divides by its length, or the team agrees that the editor sends a step as a very short ramp.
8. Long curves. `app/media.py` builds automation as nested `if()`. An expression of that shape, built by a local script from the code in `app/media.py`, rendered in ffmpeg 8.1 at 64 keyframes and failed to parse at 99. `app/media.py` itself was not run. Prompts stay under that (intensity at most 40, the agent at most 64), but keyframes added by hand can pass it, and sync sends them.

### Person 1 (audio)

1. Stems have only a `name`, with no `kind`. Intensity automation needs `dialogue` stems, and the editor does not guess kinds from names, so real stems default to `sfx` and the track header has a kind picker. A pipeline kind field would remove the manual step, but nothing depends on it. Picked kinds stay in the editor: the backend has no field for them, so they are not sent and are lost on reload.

### Person 2 (vision)

1. Nothing in the editor consumes masks or click-to-select yet. `PUT /api/projects/{id}/visual-metadata` is the documented hand-off, and it is unused here.
