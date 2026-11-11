"""VAD speech screening: find speech regions with timestamps, merge nearby ones into blocks.

Implemented on silero VAD (onnxruntime) via faster-whisper's bundled model — MIT licensed,
no torch required. This is the independent implementation of the "screen first, then
recognize only speech" core logic: a two-hour video with six minutes of dialogue runs
ASR on six minutes, not two hours.
"""
import numpy as np
import wave

VAD_SAMPLE_RATE = 16000


def _vad_model():
    # faster-whisper >= 1.1 exposes a cached silero model loader
    from faster_whisper.vad import get_vad_model
    return get_vad_model()


def _speech_regions(audio_f32, threshold=0.35):
    """[(start_sec, end_sec)] via silero onnx; audio is float32 mono at 16 kHz."""
    from faster_whisper.vad import VadOptions, get_speech_timestamps

    model = _vad_model()
    opts = VadOptions(
        threshold=threshold,
        min_speech_duration_ms=150,
        min_silence_duration_ms=400,
        speech_pad_ms=120,
    )
    ts = get_speech_timestamps(audio_f32, opts, sampling_rate=VAD_SAMPLE_RATE, model=model)
    return [(t["start"] / VAD_SAMPLE_RATE, t["end"] / VAD_SAMPLE_RATE) for t in ts]


def merge_regions(regions, max_gap, min_len):
    """Join regions separated by <= max_gap seconds; drop blocks shorter than min_len."""
    merged = []
    for s, e in regions:
        if merged and s - merged[-1][1] <= max_gap:
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e])
    return [(s, e) for s, e in merged if e - s >= min_len]


def _quietest_cut(audio, lo, hi, sr, frame=0.04):
    """Sample time inside [lo, hi) with the lowest short-window energy - the least
    damaging place to cut continuous speech in two."""
    a = max(0, int(lo * sr))
    b = min(len(audio), int(hi * sr))
    step = max(1, int(frame * sr))
    best, best_e = (a + b) // 2, None
    for i in range(a, max(a + 1, b - step), step):
        e = float(np.sum(np.square(audio[i:i + step])))
        if best_e is None or e < best_e:
            best, best_e = i + step // 2, e
    return best / sr


def split_long_blocks(blocks, audio=None, max_len=25.0, sr=VAD_SAMPLE_RATE):
    """Cap block length.

    Whisper works in 30 s windows and the batched pipeline silently transcribes
    only the first 30 s of a longer clip, so a 5-minute monologue would lose its
    tail. Cutting at the quietest point keeps every block inside one window
    without chopping a word in half whenever there is a breath to cut on.
    """
    out = []
    for s, e in blocks:
        if e - s <= max_len:
            out.append((round(s, 3), round(e, 3)))
            continue
        start = s
        while e - start > max_len:
            if audio is not None:
                cut = _quietest_cut(audio, start + max_len * 0.55, start + max_len, sr)
            else:
                cut = start + max_len
            if cut <= start + 1.0:
                cut = start + max_len
            out.append((round(start, 3), round(cut, 3)))
            start = cut
        if e - start > 0.05:
            out.append((round(start, 3), round(e, 3)))
    return out


def load_wav(path):
    """Return (float32 samples, duration_seconds)."""
    with wave.open(str(path), "rb") as w:
        sr = w.getframerate()
        n = w.getnframes()
        pcm = np.frombuffer(w.readframes(n), dtype=np.int16)
    audio = pcm.astype(np.float32) / 32768.0
    return audio, len(audio) / sr


def screen_speech(wav_path, max_gap=0.5, min_len=0.6):
    """The screening stage: returns (blocks, duration, coverage).

    Quiet murmur is inaudible to VAD at its original level, so the scan runs on a
    peak-normalized view of each chunk (up to +20 dB, never attenuated); timestamps
    are mapped back to the original timeline. A chunk the primary threshold calls
    silent is re-scanned at a fallback threshold before being skipped.
    """
    audio, duration = load_wav(wav_path)
    chunk = 10 * VAD_SAMPLE_RATE  # small chunks: a quiet-only chunk gets a real boost
    regions = []
    for c0 in range(0, len(audio), chunk):
        seg = audio[c0:c0 + chunk]
        peak = float(np.max(np.abs(seg))) if len(seg) else 0.0
        view = seg * min(10.0, 0.95 / peak) if peak > 0 else seg
        found = _speech_regions(view, threshold=0.25)
        if not found:
            found = _speech_regions(view, threshold=0.15)
        base = c0 / VAD_SAMPLE_RATE
        regions += [(s + base, e + base) for s, e in found]
    blocks = merge_regions(regions, max_gap=max_gap, min_len=min_len)
    blocks = split_long_blocks(blocks, audio=audio, max_len=25.0)
    if not blocks:
        return blocks, duration, 0.0
    speech_sec = sum(e - s for s, e in blocks)
    return blocks, duration, speech_sec / duration
