"""On-demand model store for the optional Qwen3-ASR engines.

The default build ships only whisper. Qwen weights (~1.6 GB / ~4.1 GB) are
downloaded on demand from the first source that responds, with byte-range
resume, so a dropped connection never restarts from zero.

Nothing here is Qwen- or VideoLingo-specific plumbing: plain HTTP + files.
"""
import json
import os
import threading
import time
import urllib.request

from app.settings import ROOT

_DEP_LOCK = threading.Lock()
_state = {
    "status": "idle",      # idle | running | done | error | cancelled
    "engine": None,
    "file": None,
    "done": 0,
    "total": 0,
    "pct": 0.0,
    "speed": 0,
    "started": 0.0,
    "error": None,
}
_thread = None
_cancel = threading.Event()

# weights live next to the exe / repo; SUBTITLES_MODELS_DIR overrides for tests
MODELS_DIR = os.environ.get("SUBTITLES_MODELS_DIR") or os.path.join(ROOT, "models")

MODELS = {
    "qwen-0.6b": {
        "repo": "Qwen/Qwen3-ASR-0.6B-hf",
        "dir": "Qwen3-ASR-0.6B-hf",
        "label": "Qwen3-ASR 0.6B",
        "approx_mb": 1500,
    },
    "qwen-1.7b": {
        "repo": "Qwen/Qwen3-ASR-1.7B-hf",
        "dir": "Qwen3-ASR-1.7B-hf",
        "label": "Qwen3-ASR 1.7B",
        "approx_mb": 3900,
    },
}

FILES = [
    "config.json",
    "generation_config.json",
    "processor_config.json",
    "tokenizer_config.json",
    "chat_template.jinja",
    "tokenizer.json",
    "model.safetensors",
]

SOURCES = [
    ("modelscope", "https://modelscope.cn/api/v1/models/{repo}/repo?Revision=master&FilePath={file}"),
    ("hf-mirror", "https://hf-mirror.com/{repo}/resolve/main/{file}"),
    ("huggingface", "https://huggingface.co/{repo}/resolve/main/{file}"),
]

REQUIRED = ("config.json", "model.safetensors", "tokenizer.json")


_deps_error = None


def deps_ok():
    """torch + transformers>=5.13 are needed to actually run the weights."""
    global _deps_error
    try:
        import torch  # noqa: F401
        import transformers
        from importlib.metadata import version  # noqa: F401
        if tuple(int(x) for x in transformers.__version__.split(".")[:2]) < (5, 13):
            _deps_error = f"transformers {transformers.__version__} < 5.13"
            return False
        _deps_error = None
        return True
    except Exception as e:
        _deps_error = f"{e.__class__.__name__}: {e}"
        return False


def model_dir(engine):
    return os.path.join(MODELS_DIR, MODELS[engine]["dir"])


def installed(engine):
    d = model_dir(engine)
    return all(os.path.isfile(os.path.join(d, f)) and os.path.getsize(os.path.join(d, f)) > 0
               for f in REQUIRED)


def size_mb(engine):
    d = model_dir(engine)
    total = 0
    if os.path.isdir(d):
        for f in os.listdir(d):
            p = os.path.join(d, f)
            if os.path.isfile(p):
                total += os.path.getsize(p)
    return round(total / 1048576)


def status():
    from app import runtime_pack
    return {
        "deps_ok": deps_ok(),
        "deps_error": _deps_error,
        "runtime": runtime_pack.status(),
        "models_dir": MODELS_DIR,
        "models": [
            {
                "engine": k,
                "label": v["label"],
                "installed": installed(k),
                "size_mb": size_mb(k),
                "approx_mb": v["approx_mb"],
            }
            for k, v in MODELS.items()
        ],
        "download": dict(_state),
    }


# ------------------------------------------------------------------ download

def _opener(direct=False):
    if direct:
        return urllib.request.build_opener(urllib.request.ProxyHandler({}))
    return urllib.request.build_opener()


def _request(url, start=0, timeout=30):
    headers = {"User-Agent": "Subtitles/1.0"}
    if start:
        headers["Range"] = f"bytes={start}-"
    req = urllib.request.Request(url, headers=headers)
    try:
        return _opener().open(req, timeout=timeout)
    except Exception:
        # corporate/lan proxies often break large downloads: retry direct
        return _opener(direct=True).open(req, timeout=timeout)


def _remote_size(url):
    try:
        resp = _request(url)
        n = int(resp.headers.get("Content-Length") or 0)
        resp.close()
        return n
    except Exception:
        return 0


def _download_file(url_for, dest, expected, cancel):
    part = dest + ".part"
    start = os.path.getsize(part) if os.path.isfile(part) else 0
    if expected and start >= expected:
        if os.path.isfile(dest):
            os.remove(dest)
        os.replace(part, dest)
        return start

    resp = _request(url_for(), start=start)
    mode = "ab" if start and resp.headers.get("Content-Range") else "wb"
    if mode == "wb":
        start = 0
    _state["file"] = os.path.basename(dest)
    got = start
    last = time.time()
    last_bytes = got
    chunk = 1 << 20
    with open(part, mode) as fh:
        while True:
            if cancel.is_set():
                raise _Cancelled()
            buf = resp.read(chunk)
            if not buf:
                break
            fh.write(buf)
            got += len(buf)
            _state["done"] = _state["base"] + got
            now = time.time()
            if now - last >= 0.8:
                _state["speed"] = int((got - last_bytes) / (now - last))
                last, last_bytes = now, got
                if _state["total"]:
                    _state["pct"] = round(min(100.0, _state["done"] * 100.0 / _state["total"]), 1)
    resp.close()
    if expected and os.path.getsize(part) < expected:
        raise IOError(f"incomplete download ({os.path.getsize(part)}/{expected} bytes)")
    if os.path.isfile(dest):
        os.remove(dest)
    os.replace(part, dest)
    return got


class _Cancelled(Exception):
    pass


def _run(engine):
    global _thread
    info = MODELS[engine]
    dest_dir = model_dir(engine)
    os.makedirs(dest_dir, exist_ok=True)
    try:
        sizes = {}
        total = 0
        for name in FILES:
            n = 0
            for _tag, tpl in SOURCES:
                n = _remote_size(tpl.format(repo=info["repo"], file=name))
                if n:
                    break
            sizes[name] = n
            total += n
        _state["total"] = total or info["approx_mb"] * 1048576
        for name in FILES:
            if _cancel.is_set():
                raise _Cancelled()
            _state["base"] = sum(sizes[n] for n in FILES[:FILES.index(name)])
            expected = sizes.get(name) or 0

            def url_for(name=name):
                for _tag, tpl in SOURCES:
                    url = tpl.format(repo=info["repo"], file=name)
                    try:
                        r = _request(url)
                        r.close()
                        return url
                    except Exception:
                        continue
                raise IOError(f"no reachable source for {name}")

            _download_file(url_for, os.path.join(dest_dir, name), expected, _cancel)
        _state.update(status="done", pct=100.0, speed=0)
    except _Cancelled:
        _state.update(status="cancelled", speed=0)
    except Exception as e:  # noqa: BLE001 - surfaced verbatim in the UI
        _state.update(status="error", error=f"{e.__class__.__name__}: {e}"[:300], speed=0)


def start(engine):
    global _thread
    if engine not in MODELS:
        return {"ok": False, "error": "unknown engine"}
    with _DEP_LOCK:
        if _thread and _thread.is_alive():
            return {"ok": False, "error": "a download is already running"}
        # Downloading weights is plain HTTP and needs no inference runtime: a
        # lite build may pre-fetch them (external drive, later full install).
        # The UI warns when this build can only store, not run, the engine.
        _cancel.clear()
        _state.update(status="running", engine=engine, file=None, done=0, total=0,
                      pct=0.0, speed=0, started=time.time(), error=None, base=0)
        _thread = threading.Thread(target=_run, args=(engine,), daemon=True)
        _thread.start()
    return {"ok": True}


def cancel():
    _cancel.set()
    return {"ok": True}


def delete(engine):
    import shutil
    if engine in MODELS and os.path.isdir(model_dir(engine)):
        shutil.rmtree(model_dir(engine), ignore_errors=True)
    return {"ok": True}
