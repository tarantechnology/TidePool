"""Minimal Standard MIDI File writer/reader (format 1, one tempo, one program). No dependencies."""
from __future__ import annotations

import struct
from typing import Iterable

PPQ = 480


def _vlq(n: int) -> bytes:
    out = [n & 0x7F]
    n >>= 7
    while n:
        out.append(0x80 | (n & 0x7F))
        n >>= 7
    return bytes(reversed(out))


def _track(events: list[tuple[int, bytes]]) -> bytes:
    events.sort(key=lambda e: e[0])
    body, last = b"", 0
    for tick, msg in events:
        body += _vlq(tick - last) + msg
        last = tick
    body += _vlq(0) + b"\xff\x2f\x00"
    return b"MTrk" + struct.pack(">I", len(body)) + body


def write_midi(notes: Iterable, tempo: float = 120.0, program: int = 0, channel: int = 0, name: str = "part") -> bytes:
    """Notes with ``start``/``end`` in seconds -> SMF bytes at a constant ``tempo`` (seconds are exact)."""
    us_per_beat = int(round(60_000_000 / max(20.0, tempo)))
    ticks_per_s = PPQ * 1_000_000 / us_per_beat
    meta = [(0, b"\xff\x51\x03" + us_per_beat.to_bytes(3, "big")), (0, b"\xff\x58\x04\x04\x02\x18\x08")]
    ch = channel & 0x0F
    ev: list[tuple[int, bytes]] = [(0, b"\xff\x03" + _vlq(len(name)) + name.encode()), (0, bytes([0xC0 | ch, program & 0x7F]))]
    for n in notes:
        s = int(round(n.start * ticks_per_s))
        e = max(s + 1, int(round(n.end * ticks_per_s)))
        p, v = int(n.pitch) & 0x7F, max(1, min(127, int(n.velocity)))
        ev.append((s, bytes([0x90 | ch, p, v])))
        ev.append((e, bytes([0x80 | ch, p, 0])))
    # note-offs before note-ons at the same tick (re-attacks of the same pitch)
    ev.sort(key=lambda x: (x[0], 0 if x[1][0] & 0xF0 == 0x80 else 1))
    header = b"MThd" + struct.pack(">IHHH", 6, 1, 2, PPQ)
    return header + _track(meta) + _track(ev)


def read_midi_notes(data: bytes) -> list[tuple[float, float, int, int, int]]:
    """(start_s, end_s, pitch, velocity, program) for every note; honours tempo changes."""
    if data[:4] != b"MThd":
        raise ValueError("not a MIDI file")
    _, fmt, ntrk, div = struct.unpack(">IHHH", data[4:14])
    pos = 14
    raw_events: list[tuple[int, int, bytes]] = []  # (tick, order, msg)
    order = 0
    for _ in range(ntrk):
        assert data[pos:pos + 4] == b"MTrk"
        length = struct.unpack(">I", data[pos + 4:pos + 8])[0]
        trk = data[pos + 8:pos + 8 + length]
        pos += 8 + length
        i, tick, status = 0, 0, 0
        while i < len(trk):
            delta = 0
            while True:
                b = trk[i]
                i += 1
                delta = (delta << 7) | (b & 0x7F)
                if not b & 0x80:
                    break
            tick += delta
            b = trk[i]
            if b == 0xFF:
                mtype, i = trk[i + 1], i + 2
                ln = 0
                while True:
                    c = trk[i]
                    i += 1
                    ln = (ln << 7) | (c & 0x7F)
                    if not c & 0x80:
                        break
                raw_events.append((tick, order, bytes([0xFF, mtype]) + trk[i:i + ln]))
                i += ln
            elif b in (0xF0, 0xF7):
                ln = trk[i + 1]
                i += 2 + ln
            else:
                if b & 0x80:
                    status, i = b, i + 1
                size = 1 if status & 0xF0 in (0xC0, 0xD0) else 2
                raw_events.append((tick, order, bytes([status]) + trk[i:i + size]))
                i += size
            order += 1
    raw_events.sort()
    tempo, last_tick, t_sec = 500000, 0, 0.0
    on: dict[tuple[int, int], tuple[float, int]] = {}
    prog = [0] * 16
    notes = []
    for tick, _, msg in raw_events:
        t_sec += (tick - last_tick) * tempo / 1e6 / div
        last_tick = tick
        if msg[0] == 0xFF:
            if msg[1] == 0x51:
                tempo = int.from_bytes(msg[2:5], "big")
            continue
        kind, ch = msg[0] & 0xF0, msg[0] & 0x0F
        if kind == 0xC0:
            prog[ch] = msg[1]
        elif kind == 0x90 and msg[2] > 0:
            on[(ch, msg[1])] = (t_sec, msg[2])
        elif kind in (0x80, 0x90):
            if (ch, msg[1]) in on:
                s, v = on.pop((ch, msg[1]))
                notes.append((s, t_sec, msg[1], v, 128 if ch == 9 else prog[ch]))
    return sorted(notes)
