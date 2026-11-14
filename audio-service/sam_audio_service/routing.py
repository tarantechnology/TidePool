"""Model router: which separator handles a request.

Two separators live in the service: the *prompted* model (SAM Audio or AudioSep: one text,
span or visual prompt in, one target out) and the *music stem splitter* (Demucs: a song in,
one track per instrument out). ``route`` picks one per request.

Decision table (first matching row wins; ``auto`` is the default mode):

==========================================  ===========================================
request                                     route
==========================================  ===========================================
``options.mode = prompted``                 prompted
``options.mode = music_stems``              music_stems (400 UNSUPPORTED_OPERATION when no
                                            splitter is configured, 400 INVALID_REQUEST
                                            for names that are not stems)
``options.mode = music_substems``           music_substems: the nested splitter on one stem
                                            of an earlier split (400 UNSUPPORTED_OPERATION
                                            when none is configured or ``parent_label`` is
                                            not splittable, 400 INVALID_REQUEST for names
                                            it cannot separate). Never chosen by auto.
auto, no splitter configured                prompted
auto, no prompts and no ``options.stems``   music_stems, every stem (warns when the
                                            classifier says it is not a song)
auto, a prompt has span/visual guidance     prompted (only the prompted model uses it)
auto, no prompt names a stem                prompted
auto, some prompts name stems, some not     prompted ("mixed prompts")
auto, requested stems overlap               prompted (e.g. ``instrumental`` + ``drums``)
auto, all stems, classifier says song       music_stems
auto, all stems, classifier says not song   prompted
auto, all stems, classifier unavailable     music_stems if ``hints.source_kind = audio``,
                                            otherwise prompted
==========================================  ===========================================

The last row is deliberate: a "singing" prompt on a video soundtrack must not silently
become a Demucs split, so without a classifier only an audio-only upload (the caller's
hint, the one tie-breaker) is trusted to be a song. ``options.stems``, when given, stands in
for the prompts in every row. The classifier runs at most once and only when a row needs it.
"Says song" means ``classifier.decision_score(scores) >= AUDIO_ROUTER_MUSIC_THRESHOLD``: music
with nobody talking over it, so speech over a music bed (vlogs, soundtracks) stays prompted.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from .classifier import decision_score
from .contracts import InferenceOptions, PromptSpec, RoutingHints
from .errors import InvalidRequest, UnsupportedOperation
from .stems.vocabulary import canonical_stem, components, display_name, ordered, requestable

Engine = tuple[str, str | None]  # (engine name, model id)


@dataclass(frozen=True)
class StemRequest:
    name: str  # stem name shown to the user: the caller's wording, or the title-cased canonical name
    canonical: str  # vocals | drums | bass | guitar | piano | other | instrumental


@dataclass
class RouteDecision:
    mode: str  # prompted | music_stems | music_substems
    engine: str
    model_id: str | None
    reason: str
    scores: dict[str, float] | None = None
    requests: list[StemRequest] = field(default_factory=list)  # music_stems: the target stems
    residual: list[str] = field(default_factory=list)  # music_stems: splitter stems summed into the residual
    warnings: list[str] = field(default_factory=list)
    parent: str | None = None  # music_substems: canonical label of the stem being split

    def to_dict(self) -> dict[str, Any]:
        stems = [r.canonical for r in self.requests] if self.mode == "music_stems" or (self.mode == "music_substems" and self.requests) else None
        return {"mode": self.mode, "engine": self.engine, "model_id": self.model_id, "reason": self.reason, "scores": self.scores, "stems": stems}


def match(items: list[tuple[str | None, str]], available: tuple[str, ...]) -> tuple[list[StemRequest], list[str]]:
    """Stem requests for the ``(name, wording)`` items that name a stem this splitter can
    produce, and the wordings that do not."""
    valid = set(requestable(available))
    matched: list[StemRequest] = []
    unknown: list[str] = []
    for name, wording in items:
        canonical = canonical_stem(wording)
        if canonical in valid:
            matched.append(StemRequest(name or display_name(canonical), canonical))
        else:
            unknown.append(wording)
    return matched, unknown


def select(requests: list[StemRequest], available: tuple[str, ...]) -> tuple[list[StemRequest], list[str], list[str]]:
    """Drop repeated stems and find the residual. Returns ``(requests, residual stems, warnings)``.

    Raises ValueError when two requests share a splitter stem (``instrumental`` + ``drums``):
    the returned tracks would no longer sum to the mixture.
    """
    kept: list[StemRequest] = []
    covered: set[str] = set()
    warnings: list[str] = []
    for request in requests:
        if any(k.canonical == request.canonical for k in kept):
            warnings.append(f"stem {request.name!r} repeats {request.canonical!r}; it is returned once")
            continue
        parts = set(components(request.canonical, available))
        if parts & covered:
            raise ValueError(f"{request.canonical!r} overlaps {', '.join(sorted(parts & covered))} requested by another stem")
        covered |= parts
        kept.append(request)
    return kept, [s for s in ordered(available) if s not in covered], warnings


def _items(prompts: list[PromptSpec], options: InferenceOptions) -> list[tuple[str | None, str]]:
    """``(stem name or None, wording)`` per requested stem; ``options.stems`` wins over prompts."""
    if options.stems is not None:
        return [(None, wording) for wording in options.stems]
    return [(p.display_name, p.text) for p in prompts]


def route(
    prompts: list[PromptSpec],
    options: InferenceOptions,
    hints: RoutingHints,
    *,
    available: tuple[str, ...] | None,
    classify: Callable[[], dict[str, float] | None],
    default_mode: str,
    threshold: float,
    prompted_engine: Engine,
    music_engine: Engine | None,
    substem_engine: Engine | None = None,
    substem_families: dict[str, str] | None = None,
    substem_parents: tuple[str, ...] = ("other",),
) -> RouteDecision:
    """Decide per the table in the module docstring. ``available`` is the splitter's stems
    (None when no splitter is configured); ``classify`` is called at most once.
    ``substem_families`` maps each family the nested splitter offers to its model output."""
    mode = options.mode or default_mode
    origin = "options.mode" if options.mode else "AUDIO_ROUTING_DEFAULT"
    items = _items(prompts, options)

    if mode == "music_substems":
        return _route_substems(items, options, substem_engine, substem_families, substem_parents)

    def prompted(reason: str, scores: dict[str, float] | None = None) -> RouteDecision:
        decision = RouteDecision("prompted", prompted_engine[0], prompted_engine[1], reason, scores)
        if options.stems is not None:
            decision.warnings.append("options.stems applies to music_stems only and was ignored")
        return decision

    def music(reason: str, requests: list[StemRequest], scores: dict[str, float] | None = None) -> RouteDecision:
        assert available is not None and music_engine is not None
        if not requests:
            requests = [StemRequest(display_name(s), s) for s in ordered(available)]
        kept, residual, warnings = select(requests, available)
        return RouteDecision("music_stems", music_engine[0], music_engine[1], reason, scores, kept, residual, warnings)

    if mode == "prompted":
        return prompted(f"{origin}=prompted")
    if mode == "music_stems":
        if available is None:
            raise UnsupportedOperation("music_stems is not available: no stem splitter is configured (AUDIO_STEM_SPLITTER=off)", details={"modes": ["prompted"]})
        matched, unknown = match(items, available)
        if unknown:
            raise InvalidRequest(f"not a music stem: {', '.join(repr(u) for u in unknown)}; valid stems are {', '.join(requestable(available))}", details={"unknown": unknown, "valid_stems": requestable(available)})
        try:
            decision = music(f"{origin}=music_stems", matched)
        except ValueError as exc:
            raise InvalidRequest(str(exc), details={"valid_stems": requestable(available)}) from exc
        if options.stems is None and any(p.anchors or p.visual_prompt for p in prompts):
            decision.warnings.append("music_stems ignores span anchors and visual prompts")
        return decision

    # auto
    if available is None:
        return prompted("auto: no stem splitter is configured")
    if not items:
        scores = classify()
        decision = music("auto: no prompts, so the song is split into every stem", [], scores)
        if scores is not None and decision_score(scores) < threshold:
            decision.warnings.append(f"this does not sound like music ({_describe(scores)} < {threshold:g}); split anyway because no prompts were given")
        return decision
    if options.stems is None and any(p.anchors or p.visual_prompt for p in prompts):
        return prompted("auto: span or visual guidance needs the prompted model")
    matched, unknown = match(items, available)
    if not matched:
        return prompted("auto: the prompts are not music stems")
    if unknown:
        return prompted(f"auto: mixed prompts; not music stems: {', '.join(repr(u) for u in unknown)}")
    try:
        select(matched, available)
    except ValueError as exc:
        return prompted(f"auto: {exc}")
    scores = classify()
    if scores is None:
        if hints.source_kind == "audio":
            return music("auto: the prompts are stem names and the upload is audio-only (content classifier unavailable)", matched)
        return prompted("auto: the prompts are stem names but the content classifier is unavailable and the upload is not known to be audio-only")
    if decision_score(scores) >= threshold:
        return music(f"auto: the prompts are stem names and the content is music ({_describe(scores)} >= {threshold:g})", matched, scores)
    return prompted(f"auto: the prompts are stem names but the content is not a song ({_describe(scores)} < {threshold:g})", scores)


def _route_substems(items: list[tuple[str | None, str]], options: InferenceOptions, engine: Engine | None,
                    families: dict[str, str] | None, parents: tuple[str, ...]) -> RouteDecision:
    """``music_substems``: validate the parent and the requested instruments. An empty request
    list means "detect them" (done by the engine on the audio)."""
    from .stems.nested import canonical_substem, DISPLAY_NAMES

    if engine is None or families is None:
        raise UnsupportedOperation("music_substems is not available: no nested splitter is configured (AUDIO_SUBSTEM_MODEL=off)", details={"modes": ["prompted", "music_stems"]})
    parent = (options.parent_label or "other").strip().lower()
    if parent not in parents:
        raise UnsupportedOperation(f"stem {parent!r} cannot be split further; splittable stems: {', '.join(parents)}", details={"parent_label": parent, "parents": list(parents)})
    offered = list(families)
    requests: list[StemRequest] = []
    unknown: list[str] = []
    warnings: list[str] = []
    for name, wording in items:
        family = canonical_substem(wording)
        if family not in families:
            unknown.append(wording)
            continue
        same = next((r for r in requests if families[r.canonical] == families[family]), None)
        if same is not None:
            warnings.append(f"{wording!r} is separated together with {same.name!r}; it is returned once")
            continue
        requests.append(StemRequest(name or DISPLAY_NAMES[family], family))
    if unknown:
        raise InvalidRequest(f"cannot separate {', '.join(repr(u) for u in unknown)} from this stem; this model separates {', '.join(offered)}",
                             details={"unknown": unknown, "valid_stems": offered})
    reason = f"options.mode=music_substems of {parent!r}: " + (f"requested {', '.join(r.canonical for r in requests)}" if requests else "instruments detected by the tagger")
    decision = RouteDecision("music_substems", engine[0], engine[1], reason, None, requests, [], warnings, parent)
    return decision


def _describe(scores: dict[str, float]) -> str:
    """``song score 0.52 [music 0.60, speech 0.08]`` for reasons and warnings."""
    return f"song score {decision_score(scores):.2f} [music {scores['music']:.2f}, speech {scores.get('speech', 0.0):.2f}]"
