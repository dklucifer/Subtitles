"""Build the optional Qwen runtime pack (torch + transformers) for the lite app.

Usage:
    python tools/make_runtime_pack.py [target_dir]

Default target is <repo>/dist/Subtitles/runtime, i.e. right next to the exe, so
the packaged app picks it up immediately. Everything is copied from the project
venv, so the pack matches the Python ABI the exe was built with (3.13).

Link-time artifacts are dropped: torch/lib/*.lib alone is 2.7 GB and is never
read at runtime.
"""
import os
import shutil
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VENV_SITE = os.path.join(REPO, ".venv", "Lib", "site-packages")

# import name -> extra pruning for that package
PACKAGES = {
    "torch": "torch",
    "transformers": "transformers",
    "accelerate": "accelerate",
    "huggingface_hub": "huggingface_hub",
    "tokenizers": None,          # already bundled in the exe, kept for safety
    "safetensors": None,
    "requests": "requests",
    "urllib3": "urllib3",
    "certifi": "certifi",
    "idna": "idna",
    "charset_normalizer": "charset_normalizer",
    "packaging": "packaging",
    "filelock": "filelock",
    "tqdm": "tqdm",
    "regex": "regex",
    "sympy": "sympy",
    "mpmath": "mpmath",
    "networkx": "networkx",
    "jinja2": "jinja2",
    "markupsafe": "markupsafe",
    "fsspec": "fsspec",
    "cffi": "cffi",
    "pycparser": "pycparser",
    "soundfile": "soundfile",
    "typing_extensions": "typing_extensions.py",
    "hf_xet": "hf_xet",
}

# packages that ship native bits outside their own directory
EXTRAS = ["_soundfile.py", "_soundfile_data", "_cffi_backend*.pyd"]

# directories inside torch that are never needed to run inference
# (torch.testing is NOT one of them: torch/__init__.py imports it)
TORCH_SKIP_DIRS = {"include", "test", "share"}
SKIP_SUFFIX = (".lib", ".pdb", ".exp")


def copy_tree(src, dst, skip_dirs=()):
    for root, dirs, files in os.walk(src):
        rel = os.path.relpath(root, src)
        if rel == ".":
            dirs[:] = [d for d in dirs if d not in skip_dirs and d != "__pycache__"]
        else:
            dirs[:] = [d for d in dirs if d != "__pycache__"]
        out = os.path.join(dst, rel) if rel != "." else dst
        os.makedirs(out, exist_ok=True)
        for f in files:
            if f.endswith(SKIP_SUFFIX) or f.endswith(".pyc"):
                continue
            try:
                shutil.copy2(os.path.join(root, f), os.path.join(out, f))
            except (OSError, shutil.Error):
                pass


def main():
    target = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
        REPO, "dist", "Subtitles", "runtime")
    target = os.path.abspath(target)
    if not os.path.isdir(VENV_SITE):
        sys.exit(f"venv site-packages not found: {VENV_SITE}")
    if os.path.isdir(target):
        shutil.rmtree(target)
    os.makedirs(target, exist_ok=True)

    entries = os.listdir(VENV_SITE)
    copied, missing = [], []
    for name, prune in PACKAGES.items():
        hit = None
        for e in entries:
            if e == name or e == name + ".py":
                hit = e
                break
        if not hit:
            missing.append(name)
            continue
        src = os.path.join(VENV_SITE, hit)
        dst = os.path.join(target, hit)
        if os.path.isdir(src):
            skip = TORCH_SKIP_DIRS if name == "torch" else ()
            copy_tree(src, dst, skip)
        else:
            shutil.copy2(src, dst)
        copied.append(name)
        # dist-info carries the version that importlib.metadata reports
        for e in entries:
            if e.lower().startswith(name.lower().replace("-", "_") + "-") and e.endswith(".dist-info"):
                shutil.copytree(os.path.join(VENV_SITE, e), os.path.join(target, e),
                                dirs_exist_ok=True)
                break

    total = 0
    import glob as _glob
    for pattern in EXTRAS:
        for src in _glob.glob(os.path.join(VENV_SITE, pattern)):
            dst = os.path.join(target, os.path.basename(src))
            if os.path.isdir(src):
                copy_tree(src, dst)
            else:
                shutil.copy2(src, dst)

    total = 0
    for root, _dirs, files in os.walk(target):
        for f in files:
            try:
                total += os.path.getsize(os.path.join(root, f))
            except OSError:
                pass

    print(f"target : {target}")
    print(f"size   : {total / 1e9:.2f} GB")
    print(f"copied : {', '.join(copied)}")
    if missing:
        print(f"absent : {', '.join(missing)}")


if __name__ == "__main__":
    main()
