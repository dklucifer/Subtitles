"""SRT export: source-only, translation-only or bilingual."""
import os


def _fmt_ts(seconds):
    ms = int(round(seconds * 1000))
    h, rem = divmod(ms, 3600000)
    m, rem = divmod(rem, 60000)
    s, ms = divmod(rem, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def _wrap(text, max_chars):
    text = text.strip()
    if len(text) <= max_chars:
        return text
    # split at the middle-most space/punctuation for a balanced two-line cue
    mid = len(text) // 2
    candidates = [i for i, ch in enumerate(text) if ch in " ,.;:!?、，。；：！？"]
    if candidates:
        mid = min(candidates, key=lambda i: abs(i - mid))
    return text[:mid + 1].rstrip() + "\n" + text[mid + 1:].lstrip()


def write_srt(segments, path, mode="bilingual", max_line_chars=42):
    """segments: [{"start","end","text","translation"}]; mode: source|translation|bilingual."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8-sig") as f:
        for i, seg in enumerate(segments, 1):
            src = seg.get("text", "")
            dst = seg.get("translation") or ""
            if mode == "source":
                body = _wrap(src, max_line_chars)
            elif mode == "translation":
                body = _wrap(dst or src, max_line_chars)
            else:
                body = _wrap(src, max_line_chars) + "\n" + _wrap(dst or "", max_line_chars)
            f.write(f"{i}\n{_fmt_ts(seg['start'])} --> {_fmt_ts(seg['end'])}\n{body}\n\n")
    return path
