# -*- mode: python ; coding: utf-8 -*-
"""Full build: bundles torch + transformers so the Qwen3-ASR engine works in the
packaged app. Expect a ~6-8 GB dist folder; model weights are NOT bundled - they
live in <app>/models/ (copy Qwen3-ASR-*-hf folders there).

Build:  .venv\\Scripts\\pyinstaller.exe Subtitles-full.spec --distpath dist-full --workpath build-full
"""
from PyInstaller.utils.hooks import collect_all

datas = [('app/static', 'app/static')]
binaries = []
hiddenimports = ['uvicorn.loops.auto', 'uvicorn.protocols.http.auto',
                 'uvicorn.protocols.websockets.auto', 'uvicorn.lifespan.on']
for pkg in ('faster_whisper', 'ctranslate2', 'onnxruntime', 'av', 'static_ffmpeg',
            'tokenizers', 'nvidia.cublas', 'nvidia.cudnn',
            'torch', 'torchaudio', 'transformers', 'safetensors',
            'accelerate', 'huggingface_hub'):
    try:
        tmp_ret = collect_all(pkg)
        datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]
    except Exception:
        pass

a = Analysis(
    ['launch.py'],
    pathex=[],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=['tensorflow'],
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
    upx=False,
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
    upx=False,
    upx_exclude=[],
    name='Subtitles',
)
