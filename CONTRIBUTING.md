# Contributing

The four workstreams share this repository. Start with [INTEGRATION.md](INTEGRATION.md), and coordinate changes to API contracts in the relevant issue before merging them.

1. Pick or create an issue and describe the approach if it affects another workstream.
2. Create a branch from `main`, such as `audio/sam-adapter`, `vision/tracking`, `frontend/editor`, or `backend/export`.
3. Keep work focused, run the relevant tests, and open a pull request referencing the issue (`Closes #123` when appropriate).
4. Request review from an affected teammate. Explain setup, validation, contract changes, and any remaining limitations in the PR.

Backend setup and testing are in [README.md](README.md). Do not commit credentials, uploaded media, generated exports, virtual environments, or local databases. Use `.env.example` to document configuration without real secrets.

## Workstream boundaries

- **Person 1 — Audio AI:** inference service, audio separation and processing. See the multipart request and response contract in INTEGRATION.md.
- **Person 2 — Vision:** detection, tracking and visual selections; coordinate prompt metadata with Person 1.
- **Person 3 — Frontend:** React editor, playback and editing flows using the documented API.
- **Person 4 — Backend:** orchestration, storage, processing jobs, export and messaging integration.

Agree on interface changes together; avoid silently changing a contract another workstream consumes.
