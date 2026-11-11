# -*- mode: python ; coding: utf-8 -*-
import os

from PyInstaller.utils.hooks import collect_all

datas = [('app/static', 'app/static')]
binaries = []
hiddenimports = ['uvicorn.loops.auto', 'uvicorn.protocols.http.auto', 'uvicorn.protocols.websockets.auto', 'uvicorn.lifespan.on']
tmp_ret = collect_all('faster_whisper')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]
tmp_ret = collect_all('ctranslate2')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]
tmp_ret = collect_all('onnxruntime')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]
tmp_ret = collect_all('av')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]
tmp_ret = collect_all('static_ffmpeg')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]
tmp_ret = collect_all('tokenizers')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]
tmp_ret = collect_all('nvidia.cublas')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]
tmp_ret = collect_all('nvidia.cudnn')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]

# Pure-python support libraries shared by faster-whisper and the optional Qwen
# runtime pack. PyInstaller only records the submodules it saw at build time,
# and frozen modules shadow anything on sys.path - so an optional pack could
# never supply the missing pieces (transformers needs e.g. jinja2.meta and the
# full huggingface_hub). Collect them completely into the bundle instead; they
# are small, and this keeps the lite build self-sufficient.
for _pkg in ('huggingface_hub', 'jinja2', 'markupsafe', 'packaging', 'tqdm',
             'requests', 'urllib3', 'certifi', 'idna', 'charset_normalizer',
             'fsspec', 'filelock', 'sympy', 'safetensors'):
    tmp_ret = collect_all(_pkg)
    datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]

# The optional Qwen runtime imports stdlib modules the app itself never touches
# (torch pulls in pickletools, unittest.mock, xml, ...). Frozen builds only ship
# what the analysis saw, so pull in the standard library wholesale - it is only
# a few MB and it makes the external runtime behave like a normal interpreter.
import sysconfig
import pathlib
from PyInstaller.utils.hooks import collect_submodules

_stdlib = pathlib.Path(sysconfig.get_paths()["stdlib"])
_skip_top = {"test", "idlelib", "lib2to3", "tkinter", "turtledemo", "ensurepip",
             "venv", "distutils", "site-packages", "antigravity", "this"}
for _p in _stdlib.glob("*.py"):
    if _p.stem.isidentifier() and _p.stem not in _skip_top:
        hiddenimports.append(_p.stem)
for _d in _stdlib.iterdir():
    if _d.is_dir() and (_d / "__init__.py").exists() and _d.name not in _skip_top:
        hiddenimports.append(_d.name)
for _m in ("unittest", "xml", "email", "http", "urllib", "importlib",
           "multiprocessing", "concurrent", "ctypes", "logging", "asyncio",
           "json", "html", "encodings"):
    try:
        hiddenimports += collect_submodules(_m)
    except Exception:
        pass

# importlib.metadata needs <dist>-<ver>.dist-info on sys.path; the versions above
# are checked at import time by transformers (pyyaml, tokenizers, ...).
import glob as _glob
import sys as _sys
_site = os.path.join(_sys.prefix, 'Lib', 'site-packages')
for _name in ('PyYAML', 'safetensors', 'filelock', 'fsspec', 'sympy', 'Jinja2',
              'MarkupSafe', 'packaging', 'tqdm', 'requests', 'urllib3', 'certifi',
              'idna', 'charset_normalizer', 'huggingface_hub', 'tokenizers', 'numpy'):
    for _d in _glob.glob(os.path.join(_site, _name + '-*.dist-info')):
        datas.append((_d, os.path.basename(_d)))


a = Analysis(
    ['launch.py'],
    pathex=[],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    # Lite build: whisper engine only (~800 MB). The optional Qwen3-ASR engine
    # lives in Subtitles-full.spec (torch/transformers add ~7 GB).
    excludes=['torch', 'transformers', 'accelerate', 'soundfile', 'tensorflow'],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='Subtitles',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='Subtitles',
)
