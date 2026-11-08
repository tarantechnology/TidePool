"""SoundCanvas Part 2: detection, tracking, selection and CV prompt preparation."""
from .pipeline import Detection, process_media, pick_object_at_point
from .selection import select_at_point
__all__ = ["Detection", "process_media", "pick_object_at_point", "select_at_point"]
