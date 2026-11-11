"""ASR stage: faster-whisper (MIT).

Speed strategy (adaptive, chosen automatically per machine):
- Device auto -> CUDA when available, else CPU.
- Compute auto -> GPU: int8 (fast on Pascal and newer; ~2x vs float16 on GTX 16xx),
  CPU: int8. `asr.compute` in config.json can override (int8 / int8_float16 / float16).
- Batch mode (GPU only) -> the screened speech blocks are concatenated into one
  compact track, BatchedInferencePipeline processes it in parallel chunks (4-8x),
  then word timestamps are mapped back onto the original timeline piece by piece.
"""
import threading

_MODEL_LOCK = threading.Lock()
_MODEL_CACHE = {}
# which decoding path actually ran last, and why it fell back (for the job log)
LAST_MODE = {"path": "per-block", "error": None}


def _patch_pyav():
    """faster-whisper passes metadata_errors=... to av.open; PyAV 15+ removed the
    parameter. Strip it so newer PyAV wheels (Python 3.13) keep working."""
    try:
        import av
        if getattr(av, "_subtitles_patched", False):
            return
        orig_open = av.open

        def open_compat(*args, **kwargs):
            kwargs.pop("metadata_errors", None)
            return orig_open(*args, **kwargs)

        av.open = open_compat
        av._subtitles_patched = True
    except Exception:
        pass


_patch_pyav()


def _resolve_compute(dev, override):
    """Adaptive compute type. int8 is the fastest safe default on both Pascal-era
    GPUs and CPU; int8_float16 only wins on tensor-core GPUs (Turing+), where the
    user can opt in explicitly via config."""
    if override and override != "auto":
        return override
    return "int8"


def _resolve_device(device, compute="auto"):
    if device == "cpu":
        return "cpu", _resolve_compute("cpu", compute)
    if device in ("cuda", "gpu"):
        return "cuda", _resolve_compute("cuda", compute)
    # auto: try cuda, fall back to cpu int8
    try:
        import ctranslate2
        if ctranslate2.get_cuda_device_count() > 0:
            _patch_cuda_dlls()
            return "cuda", _resolve_compute("cuda", compute)
    except Exception:
        pass
    return "cpu", _resolve_compute("cpu", compute)


def _patch_cuda_dlls():
    """Make cublas/cudnn wheels visible to CTranslate2 (Windows)."""
    import os
    import sys
    dirs = []
    if getattr(sys, "frozen", False):
        base = sys._MEIPASS
        dirs = [os.path.join(base, "nvidia", pkg, "bin")
                for pkg in ("cublas", "cudnn")]
    else:
        import site
        for site_dir in site.getsitepackages():
            dirs += [os.path.join(site_dir, "nvidia", name, "bin")
                     for name in ("cublas", "cudnn")]
    for p in dirs:
        if os.path.isdir(p):
            os.add_dll_directory(p)
            os.environ["PATH"] = p + os.pathsep + os.environ.get("PATH", "")


def get_model(name, device, compute="auto"):
    key = (name, device, compute)
    with _MODEL_LOCK:
        if key not in _MODEL_CACHE:
            from faster_whisper import WhisperModel
            dev, comp = _resolve_device(device, compute)
            _MODEL_CACHE[key] = (WhisperModel(name, device=dev, compute_type=comp), dev)
    return _MODEL_CACHE[key]


class CancelledError(Exception):
    pass


# ---------------------------------------------------------------- public API

def transcribe_blocks(wav_path, blocks, model_name="small", device="auto",
                      language=None, initial_prompt=None, hotwords=None,
                      min_confidence=0.0, batch=True, compute="auto",
                      on_progress=None, cancel=None):
    """Transcribe merged speech blocks. Returns (segments, resolved_device).

    segments: [{"start","end","text","words","low_confidence"}] on the original
    timeline. `min_confidence` > 0 flags segments whose avg_logprob falls below
    it (0 or None disables the check).
    """
    global LAST_MODE
    model, resolved_dev = get_model(model_name, device, compute)
    audio = _load_audio(wav_path)

    use_batch = batch and resolved_dev == "cuda" and len(blocks) > 2
    if use_batch:
        try:
            segs = _transcribe_batched(model, audio, blocks, language, hotwords,
                                       min_confidence, on_progress, cancel)
            LAST_MODE = {"path": "batch", "error": None}
            return segs, f"{resolved_dev} · batch"
        except CancelledError:
            raise
        except Exception as e:  # noqa: BLE001 - any incompatibility -> proven path
            LAST_MODE = {"path": "per-block", "error": f"{e.__class__.__name__}: {e}"[:200]}
    else:
        LAST_MODE = {"path": "per-block", "error": None}
    segs = _transcribe_per_block(model, audio, blocks, language, initial_prompt,
                                 hotwords, min_confidence, on_progress, cancel)
    return segs, f"{resolved_dev} · per-block"


def _load_audio(wav_path):
    import wave

    import numpy as np
    with wave.open(str(wav_path), "rb") as w:
        sr = w.getframerate()
        pcm = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
    return pcm.astype(np.float32) / 32768.0


def _segment_out(seg, offset, min_confidence):
    text = seg.text.strip()
    if not text:
        return None
    out = {
        "start": round(seg.start + offset, 3),
        "end": round(seg.end + offset, 3),
        "text": text,
        "words": [{"word": w.word.strip(), "start": (w.start + offset) if w.start is not None else None,
                   "end": (w.end + offset) if w.end is not None else None}
                  for w in (seg.words or [])],
    }
    if min_confidence and getattr(seg, "avg_logprob", None) is not None:
        out["low_confidence"] = seg.avg_logprob < min_confidence
    return out


# ---------------------------------------------------- path 1: batched (GPU)

# a cue longer than this is hard to read; split it at word boundaries
_MAX_CUE_CHARS = 42
_MAX_CUE_SEC = 6.0
_SENT_END = ("。", "！", "？", "!", "?", "…", "．", ".")

def _group_clips(blocks, max_sec=27.0, max_gap=2.0):
    """Join consecutive blocks into clips short enough for one Whisper window.

    Blocks come out of VAD screening anywhere from 0.6 s to several minutes. A
    clip longer than 30 s is silently truncated by the batched pipeline (it only
    reads the first window), which used to lose whole minutes of dialogue.
    Joining neighbouring blocks also gives the model context and keeps the batch
    items big enough to keep the GPU busy.
    """
    clips, cur = [], None
    for s, e in blocks:
        if cur and (e - cur[0] <= max_sec) and (s - cur[1] <= max_gap):
            cur = [cur[0], e]
        else:
            if cur:
                clips.append((cur[0], cur[1]))
            cur = [s, e]
    if cur:
        clips.append((cur[0], cur[1]))
    return clips


def _looks_repetitive(text):
    """Whisper sometimes loops ('ああ、ああ、ああ…'); drop those cues."""
    if len(text) < 12:
        return False
    for n in (2, 3, 4, 6):
        head = text[:min(len(text), 8 * n)]
        for i in range(0, max(1, len(head) - n)):
            pat = head[i:i + n]
            if pat.strip() and text.count(pat) >= 6 and len(pat) * text.count(pat) > len(text) * 0.6:
                return True
    return False


def _cues_from_words(words, max_chars=_MAX_CUE_CHARS, max_sec=_MAX_CUE_SEC, min_sec=0.16):
    """Split one Whisper segment into readable cues using real word timings."""
    cues, cur = [], []

    def emit(ws):
        text = "".join(w["word"] for w in ws).strip()
        if not text or _looks_repetitive(text):
            return
        start, end = ws[0]["start"], ws[-1]["end"]
        if end - start < min_sec:
            return                     # zero-length artefacts carry no timing
        if cues and start < cues[-1]["end"]:
            start = cues[-1]["end"]    # never go backwards on the timeline
        if end - start < min_sec:
            return
        cues.append({"start": round(start, 3), "end": round(end, 3), "text": text, "words": ws})

    for w in words:
        if w["end"] <= w["start"]:
            continue
        cur.append(w)
        span = w["end"] - cur[0]["start"]
        chars = sum(len(x["word"]) for x in cur)
        if span <= max_sec and chars <= max_chars:
            continue
        # over budget: prefer cutting at punctuation or a real pause near the end
        cut = None
        for i in range(len(cur) - 2, max(0, len(cur) // 3) - 1, -1):
            nxt_gap = cur[i + 1]["start"] - cur[i]["end"]
            if cur[i]["word"].rstrip().endswith(_SENT_END) or nxt_gap > 0.45:
                cut = i + 1
                break
        if cut:
            emit(cur[:cut])
            cur = cur[cut:]
        else:
            emit(cur)
            cur = []
    emit(cur)
    return cues


def _transcribe_batched(model, audio, blocks, language, hotwords,
                        min_confidence, on_progress, cancel):
    """Batched decoding over short clips.

    Each clip becomes one batch item, so faster-whisper returns timestamps that
    already sit on the original timeline (it offsets by the clip start). Batches
    of 8 give ~15x realtime on a GTX 1660; segments are then re-cut into readable
    cues with the word timings the pipeline computed.
    """
    import bisect

    from faster_whisper import BatchedInferencePipeline

    clips = _group_clips(blocks)
    if not clips:
        return []
    ends = [c[1] for c in clips]
    total = len(clips)

    pipe = BatchedInferencePipeline(model=model)
    kwargs = dict(
        language=language or None,
        word_timestamps=True,
        clip_timestamps=[{"start": s, "end": e} for s, e in clips],
        batch_size=8,
        beam_size=1,
        condition_on_previous_text=False,   # clips are independent
        compression_ratio_threshold=2.3,    # catches degenerate loops early
        log_prob_threshold=None,            # keep quiet/murmured speech
    )
    if hotwords:
        kwargs["hotwords"] = hotwords
    seg_iter, _info = pipe.transcribe(audio, **kwargs)

    segments = []
    reported = 0
    for seg in seg_iter:
        if cancel is not None and cancel.is_set():
            raise CancelledError()
        text = (seg.text or "").strip()
        if not text:
            continue
        # Whisper loves to invent "Thank you." / "ご視聴ありがとう" on music or
        # near-silence; both signals must be bad before a cue is thrown away.
        if getattr(seg, "no_speech_prob", 0.0) > 0.6 and getattr(seg, "avg_logprob", 0.0) < -0.85:
            continue
        if _looks_repetitive(text):
            continue

        words = [{"word": w.word, "start": round(w.start, 3), "end": round(w.end, 3)}
                 for w in (seg.words or []) if w.start is not None and w.end is not None]
        cues = _cues_from_words(words) if words else [{
            "start": round(seg.start, 3), "end": round(seg.end, 3), "text": text, "words": []}]
        for cue in cues:
            if min_confidence and getattr(seg, "avg_logprob", None) is not None:
                cue["low_confidence"] = seg.avg_logprob < min_confidence
            segments.append(cue)

        done = min(total, bisect.bisect_left(ends, seg.start) + 1)
        if on_progress and done > reported:
            reported = done
            on_progress(done, total)
    if on_progress:
        on_progress(total, total)
    return segments


# ------------------------------------------------- path 2: per block (CPU / fallback)

def _transcribe_per_block(model, audio, blocks, language, initial_prompt,
                          hotwords, min_confidence, on_progress, cancel):
    sr = 16000
    segments = []
    done = 0
    for s, e in blocks:
        if cancel is not None and cancel.is_set():
            raise CancelledError()
        chunk = audio[int(s * sr):int(e * sr)]
        segs, _ = model.transcribe(
            chunk, language=language or None,
            word_timestamps=True,
            vad_filter=False,           # we already screened
            condition_on_previous_text=False,  # blocks are independent clips
            initial_prompt=initial_prompt,
            hotwords=hotwords or None,
            beam_size=5,
        )
        for seg in segs:
            out = _segment_out(seg, s, min_confidence)
            if out:
                segments.append(out)
        done += 1
        if on_progress:
            on_progress(done, len(blocks))
    return segments
