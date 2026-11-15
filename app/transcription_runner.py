"""Isolated local faster-whisper runner. Transcript travels over private pipes, not logs."""
import json
import sys


def main():
    request = json.load(sys.stdin)
    from faster_whisper import WhisperModel
    model = WhisperModel(request['model'], device=request['device'], compute_type=request['compute'], local_files_only=True)
    segments, info = model.transcribe(request['path'], word_timestamps=True, vad_filter=True, beam_size=5, condition_on_previous_text=False)
    words = [{'word':str(w.word).strip(),'start':w.start,'end':w.end} for segment in segments for w in (segment.words or [])]
    json.dump({'words':words,'language':info.language}, sys.stdout)


if __name__ == '__main__': main()
