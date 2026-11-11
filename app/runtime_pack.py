"""Optional runtime pack loader (Qwen3-ASR support for the lite build).

The shipped build bundles only what whisper needs. The Qwen engines additionally
need torch + transformers (~4 GB). Instead of forcing every user to install a
7 GB build, those packages may live in a `runtime/` folder next to the exe:

    Subtitles/
      Subtitles.exe
      runtime/            <- optional, dropped in by the user
        torch/
        transformers/
        ...

`activate()` puts that folder *after* the bundled modules on sys.path (so the
pack can never shadow the app's own dependencies) and registers its DLL
directories, which Windows needs since Python 3.8 ignores PATH for extension
modules. Nothing here imports torch: it only prepares the search path.
"""
import os
import sys

from app.settings import ROOT

RUNTIME_DIR = os.environ.get("SUBTITLES_RUNTIME_DIR") or os.path.join(ROOT, "runtime")

_active = False


def _dll_dirs(base):
    """Every directory in the pack that may hold native libraries."""
    found = [base]
    for sub in ("torch/lib", "torch/bin", "nvidia"):
        p = os.path.join(base, sub)
        if os.path.isdir(p):
            found.append(p)
    # nvidia/<component>/bin layout used by the nvidia-* wheels
    nv = os.path.join(base, "nvidia")
    if os.path.isdir(nv):
        for comp in os.listdir(nv):
            for sub in ("bin", "lib"):
                p = os.path.join(nv, comp, sub)
                if os.path.isdir(p):
                    found.append(p)
    return found


def activate():
    """Make an optional runtime pack importable. Returns True when present."""
    global _active
    if _active:
        return True
    if not os.path.isdir(RUNTIME_DIR):
        return False
    if RUNTIME_DIR not in sys.path:
        # append, never insert: bundled modules must keep priority
        sys.path.append(RUNTIME_DIR)
    if hasattr(os, "add_dll_directory"):
        for d in _dll_dirs(RUNTIME_DIR):
            try:
                os.add_dll_directory(d)
            except OSError:
                pass
    _active = True
    return True


def status():
    """Small dict for the UI / API."""
    return {"present": os.path.isdir(RUNTIME_DIR), "path": RUNTIME_DIR, "active": _active}
