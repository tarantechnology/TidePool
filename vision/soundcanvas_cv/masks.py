"""Portable binary RLE: row-major, alternating zero/one runs, starts with zeros."""
import numpy as np


def encode_mask(mask):
    mask = np.asarray(mask)
    if mask.ndim != 2 or not mask.size or not np.isin(mask, [0, 1]).all():
        raise ValueError("Mask must be a binary HxW array")
    flat = mask.astype(np.uint8).ravel()
    edges = np.flatnonzero(np.diff(flat)) + 1
    counts = np.diff(np.r_[0, edges, flat.size]).tolist()
    if flat.size and flat[0]:
        counts.insert(0, 0)
    return {"encoding": "rle", "order": "C", "size": list(mask.shape),
            "counts": counts, "foreground_value": 1}


def decode_mask(value):
    if value.get("encoding") != "rle" or value.get("order") != "C" or value.get("foreground_value") != 1:
        raise ValueError("Unsupported mask encoding")
    h, w = value["size"]
    counts = value["counts"]
    if (type(h) is not int or type(w) is not int or h <= 0 or w <= 0 or
            not counts or any(type(c) is not int or c < 0 for c in counts) or sum(counts) != h*w):
        raise ValueError("Invalid mask size or run counts")
    return np.repeat(np.arange(len(counts), dtype=np.uint8) % 2, counts).reshape(h, w)
