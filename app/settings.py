"""Settings: a single config.json next to the app. The API key never leaves this machine."""
import json
import os
import sys

if getattr(sys, "frozen", False):
    ROOT = os.path.dirname(sys.executable)      # exe distribution: data lives beside the exe
else:
    ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BUNDLE_DIR = getattr(sys, "_MEIPASS", ROOT)      # PyInstaller one-dir payload
CONFIG_PATH = os.path.join(ROOT, "config.json")

DEFAULTS = {
    "llm": {
        "base_url": "https://api.deepseek.com/v1",
        "api_key": "",
        "model": "deepseek-chat",
    },
    "asr": {
        # tiny / base / small / medium / large-v3-turbo
        "model": "small",
        "device": "auto",        # auto | cuda | cpu
        "compute": "auto",       # auto | int8 | int8_float16 | float16 (override)
        "batch": True,           # batched inference on GPU (4-8x faster)
        "engine": "whisper",     # whisper | qwen-0.6b | qwen-1.7b (BGM-robust)
        "language": "auto",      # source language, auto = detect
        "hotwords": "",          # comma-separated boost terms (optional)
        "min_confidence": 0.0,   # avg_logprob below this -> flagged (0 = off)
    },
    "subtitle": {
        "target_language": "简体中文",
        "bilingual": True,       # burn "源文\n译文" pairs into SRT
        "max_line_chars": 42,
        "min_block_gap": 0.5,    # speech regions closer than this are merged (seconds)
        "min_block_len": 0.6,    # regions shorter than this are dropped (seconds)
        "glossary": "",          # one "source=target" per line (optional)
        "two_pass": False,       # premium: draft + polish translation pass
    },
    "server": {"port": 8765},
}


def _deep_merge(base, extra):
    out = dict(base)
    for k, v in (extra or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def load():
    cfg = DEFAULTS
    if os.path.isfile(CONFIG_PATH):
        try:
            with open(CONFIG_PATH, "r", encoding="utf-8") as f:
                cfg = _deep_merge(DEFAULTS, json.load(f))
        except Exception:
            pass
    return cfg


def save(cfg):
    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)
