# SoundCanvas editor

The interactive editor for SoundCanvas: a multitrack timeline, gain automation and prompt-driven edits in the browser. It runs with synthetic mock stems and no backend. Built with Vite, React, TypeScript and Tailwind.

## Run

```sh
npm install
npm run dev
```

Open http://localhost:5173/#/app for the editor. The landing page is at `#/`.

| Command | What it does |
|---|---|
| `npm run dev` | Dev server |
| `npm test` | Unit tests. The render tests also need `ffmpeg` on PATH, and skip without it |
| `npm run lint` | oxlint |
| `npm run build` | Type-check and production build |
| `npm run verify` | Type-check, lint, tests and build. Run this before committing |

## Using the editor

- **Timeline:** double-click a lane to add a keyframe, drag a marker to move it, and Alt-click a marker to delete it.
- **Undo:** Cmd/Ctrl+Z, and Cmd/Ctrl+Shift+Z to redo.
- **Media:** drop audio files or use **Import audio** to replace the stems. Load a video from the video panel, or drop it. The video is muted and follows the audio clock.
- **Prompts:** "Lower the siren volume to 30%" sets the fader, "Lower the siren by 20%" is relative, and adding "from 8 to 16 seconds" limits either to that range. Amounts it cannot read exactly (dB, "twice") get a question, not a guess. Also try "Make the siren louder whenever the argument gets intense". That follows how loud the dialogue is. It measures loudness, it does not detect emotion, and it needs stems of kind `dialogue`. Real stems start as SFX, so pick the kind from the dropdown in the track header first.

## Environment variables

All optional. Set `VITE_` variables in `.env.local`, and the others in your shell. Never put an API key in a `VITE_` variable, because those are compiled into the browser bundle.

| Variable | Used by | Purpose |
|---|---|---|
| `VITE_AGENT_URL` | Browser | Agent server URL. Without it, prompts use the built-in rule-based parser |
| `VITE_API_BASE_URL` | Browser | Backend base URL, or `/api` with the dev proxy. Setting it shows the job card |
| `ANTHROPIC_API_KEY` | Agent server | API key. Server side only |
| `AGENT_MODEL` | Agent server | Model ID to call |
| `AGENT_PORT` | Agent server | Port. Defaults to 8787 |
| `BACKEND_URL` | Dev server, tests | Target of the dev proxy, defaulting to http://localhost:8000. Also enables `server/backend-contract.test.ts` |

To use the LLM agent, start the server, then set `VITE_AGENT_URL` to its `/agent` URL:

```sh
node server/agent-server.mjs
```

It refuses to start without `ANTHROPIC_API_KEY` and `AGENT_MODEL`, and only accepts browser requests from http://localhost:5173. It has not been run against the live API yet.

To reach the backend from the browser, use one of these:

- **Dev proxy:** set `VITE_API_BASE_URL=/api`. The dev server forwards `/api` to `BACKEND_URL`, unchanged. `BACKEND_URL` has to be in the shell, since Vite does not put `.env` values into its own config. This works on the dev server only, not for a deployed build.
- **Direct:** set `VITE_API_BASE_URL` to the backend URL. Person 4 must add `CORSMiddleware` allowing http://localhost:5173 first, because the backend has none and browser calls with an Authorization header fail preflight without it.

With a job loaded, the editor starts from the project's saved state and sends edits to the backend as you make them. If saved state cannot be read, the stems play but nothing is sent, and the status chip says why. Stem offsets and mute ranges from the backend play back as they export, and show on the timeline as read-only. See [docs/CONTRACTS.md](docs/CONTRACTS.md).

## Documentation

- [docs/CONTRACTS.md](docs/CONTRACTS.md): shared types, API contracts and open questions for each workstream.

The backend is at the root of this repository. Outside the dev proxy, the browser cannot reach it until CORS is enabled there (see `docs/CONTRACTS.md`).
