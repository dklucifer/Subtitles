"""Qwen3-ASR engine - INDEPENDENT implementation for Subtitles.

Legal boundary (GPL isolation):
- Model weights: Qwen/Qwen3-ASR-{0.6B,1.7B}-hf are Apache-2.0, free for
  commercial use, attribution only.
- This module is written from scratch against the OFFICIAL transformers-native
  API (transformers >= 5.13, AutoProcessor / AutoModelForMultimodalLM). It does
  NOT import, read, or derive from any GPL-licensed application code (e.g.
  VideoLingo). The Subtitles project therefore stays GPL-clean.

Why this engine exists: Qwen3-ASR is notably robust against background music
and noisy mixes, where Whisper-family models degrade. Offered as an alternative
engine; the whisper path (faster-whisper, MIT) stays the default.
"""
import threading

_QWEN_LOCK = threading.Lock()
_QWEN_CACHE = {}

# engine name -> local directory under <root>/models (weights vendored there)
MODEL_DIRS = {
    "qwen-0.6b": "Qwen3-ASR-0.6B-hf",
    "qwen-1.7b": "Qwen3-ASR-1.7B-hf",
}

# whisper-style ISO codes -> language names Qwen3-ASR understands
LANG_NAMES = {
    "zh": "Chinese", "en": "English", "ja": "Japanese", "ko": "Korean",
    "de": "German", "fr": "French", "es": "Spanish", "ru": "Russian",
    "yue": "Cantonese", "pt": "Portuguese", "it": "Italian", "th": "Thai",
    "vi": "Vietnamese", "ar": "Arabic", "hi": "Hindi", "id": "Indonesian",
    "ms": "Malay", "nl": "Dutch", "sv": "Swedish", "da": "Danish",
    "fi": "Finnish", "pl": "Polish", "cs": "Czech", "tr": "Turkish",
}


def available():
    """True when torch + transformers are importable and at least one local
    model directory exists."""
    from app import model_store
    if not model_store.deps_ok():
        return False
    return any(model_store.installed(e) for e in MODEL_DIRS)


def is_ready(engine):
    """This specific engine can run right now (runtime + its own weights)."""
    from app import model_store
    return model_store.deps_ok() and model_store.installed(engine)


def _model_dir(engine):
    from app import model_store
    return model_store.model_dir(engine)


def get_model(engine="qwen-0.6b", device="auto"):
    key = (engine, device)
    with _QWEN_LOCK:
        if key not in _QWEN_CACHE:
            import torch
            from transformers import AutoModelForMultimodalLM, AutoProcessor

            path = _model_dir(engine)
            if device == "cpu":
                dev, dtype = "cpu", torch.float32
            else:
                dev = "cuda" if torch.cuda.is_available() else "cpu"
                dtype = torch.float16 if dev == "cuda" else torch.float32
            processor = AutoProcessor.from_pretrained(path)
            model = AutoModelForMultimodalLM.from_pretrained(path, dtype=dtype)
            model = model.to(dev)
            model.eval()
            _QWEN_CACHE[key] = (processor, model, dev)
    return _QWEN_CACHE[key]


def transcribe_blocks(wav_path, blocks, engine="qwen-0.6b", device="auto",
                      language=None, hotwords=None,
                      on_progress=None, cancel=None):
    """Qwen3-ASR over the screened speech blocks.

    Qwen3-ASR emits text without word timestamps, so each (sub-)chunk becomes one
    segment bounded by the screened block's own start/end — timestamps remain
    accurate because boundaries come from the VAD screening stage, not the ASR.
    Returns (segments, resolved_device_description).
    """
    import numpy as np

    from app.pipeline.vad import load_wav

    audio, _duration = load_wav(wav_path)
    sr = 16000
    processor, model, dev = get_model(engine, device)
    lang = LANG_NAMES.get(language) if language else None

    # split long blocks so generate() stays bounded
    sub_blocks = []
    for s, e in blocks:
        t = s
        while e - t > 30.0:
            sub_blocks.append((t, t + 28.0))
            t += 28.0
        sub_blocks.append((t, e))

    segments = []
    done = 0
    for s, e in sub_blocks:
        if cancel is not None and cancel.is_set():
            from app.pipeline.asr import CancelledError
            raise CancelledError()
        chunk = audio[int(s * sr):int(e * sr)]
        text = _recognize(processor, model, chunk, sr, lang, hotwords)
        segments.extend(_split_to_cues(text, s, e))
        done += 1
        if on_progress:
            on_progress(done, len(sub_blocks))
    return segments, f"{dev} (qwen {engine.split('-')[1]})"


def _split_to_cues(text, start, end):
    """Qwen3-ASR returns text without word timestamps. Split it into subtitle
    cues at sentence punctuation and distribute the block's duration by
    character count - cue boundaries stay inside the screened speech block, so
    they never drift onto silence."""
    import re

    if not text:
        return []
    parts = [p.strip() for p in re.split(r"(?<=[。！？!?；;…])\s*", text) if p.strip()]
    # break over-long sentences at commas so no cue becomes a wall of text
    cues = []
    for p in parts:
        while len(p) > 48:
            cut = max(p.rfind("，", 0, 48), p.rfind(",", 0, 48), p.rfind("、", 0, 48))
            if cut < 12:
                break
            cues.append(p[:cut + 1])
            p = p[cut + 1:].strip()
        if p:
            cues.append(p)
    if not cues:
        return []
    total_chars = sum(len(c) for c in cues) or 1
    span = max(end - start, 0.5)
    out, cursor = [], start
    for c in cues:
        dur = span * len(c) / total_chars
        out.append({"start": round(cursor, 3), "end": round(min(end, cursor + dur), 3),
                    "text": c, "words": []})
        cursor += dur
    return out


def _recognize(processor, model, audio_f32, sr, language, hotwords):
    import torch

    # audio is float32 mono in [-1, 1]; the feature extractor is configured for
    # 16 kHz, which is exactly what audio.py produces.
    kwargs = dict(audio=audio_f32)
    if language:
        kwargs["language"] = language
    if hotwords:
        # Qwen3-ASR takes domain terms as a prompt/context hint
        kwargs["prompt"] = "Keywords: " + hotwords
    inputs = processor.apply_transcription_request(**kwargs)
    inputs = inputs.to(model.device, model.dtype)
    n_sec = max(1.0, len(audio_f32) / sr)
    with torch.inference_mode():
        out = model.generate(**inputs, max_new_tokens=int(n_sec * 12) + 32)
    gen = out[:, inputs["input_ids"].shape[1]:]
    try:
        parsed = processor.decode(gen, return_format="transcription_only")[0]
    except TypeError:
        parsed = (processor.decode(gen)[0] or "").strip()
    return (parsed or "").strip()
