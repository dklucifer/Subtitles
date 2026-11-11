<div align="center">

# Subtitles

**Turn any video or audio into high-quality subtitles — local GPU speech recognition + LLM translation, fully visual.**

[![Python](https://img.shields.io/badge/Python-3.13+-blue?logo=python&logoColor=white)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Platform](https://img.shields.io/badge/Platform-Windows-lightgrey?logo=windows)](#)

[English](#-features) · [中文说明](#-中文说明)

*Screen the audio first, transcribe only the speech, then translate — no wasted GPU time on silent footage.*

</div>

---

## ✨ Features

- 🎯 **Smart speech screening** — silero VAD scans every chunk; quiet passages are re-checked with gain boost, so only real speech reaches the ASR engine
- ⚡ **Batched inference** — adjacent speech chunks are merged and transcribed in batches; measured at **6.9x realtime** on long videos
- 🧠 **Two ASR engines** — faster-whisper (default, MIT) / Qwen3-ASR (more robust against BGM and noise, downloaded on demand)
- 🌐 **Any LLM for translation** — presets for DeepSeek / OpenAI GPT / Claude / Gemini / OpenRouter / Groq / local Ollama; context bridging, glossary and optional two-pass polish
- 📝 **Readable segmentation** — word-level timestamps split by punctuation and pauses into ≤6s cues instead of 30-second walls of text
- 🖥️ **Web UI** — drag & drop upload, multi-stage progress with ETA, subtitle preview, English/Chinese interface
- 📦 **On-demand distribution** — model weights are never bundled; the built-in download page supports resume, cancel and delete

## 🚀 Quick start

### Requirements

| Item | Requirement |
|---|---|
| OS | Windows 10/11 (other platforms need ffmpeg path tweaks) |
| Python | 3.13+ |
| GPU | NVIDIA CUDA (optional — CPU works, just slower) |

### Install & run

```bash
git clone https://github.com/dklucifer/Subtitles.git
cd Subtitles
python -m venv .venv
.venv\Scripts\pip install -r requirements.txt
copy config.example.json config.json   # paste your translation API key
run.bat
```

Open <http://127.0.0.1:8765> in your browser.

> 🔑 Keys stay in your local `config.json` and never enter the repository.

## ⚙️ Pipeline (5 stages, fully visual)

```
video ─▶ 1 extract audio ─▶ 2 speech screening ─▶ 3 ASR ─▶ 4 LLM translate ─▶ 5 export SRT / burn-in
```

| Stage | What happens |
|---|---|
| 1 Extract audio | Bundled ffmpeg via static_ffmpeg — nothing to install |
| 2 Speech screening | silero VAD scans 10s chunks with peak normalization; silent chunks get a low-threshold second pass; coverage is reported |
| 3 Speech recognition | faster-whisper transcribes only speech chunks, word timestamps re-anchored to the original timeline; Qwen3-ASR optional |
| 4 Translation | Context bridging across batches, glossary-locked terms, optional two-pass (draft → polish) |
| 5 Export | Source / target / bilingual SRT, plus optional ffmpeg libass burn-in |

## 🧩 Optional helpers

- **Hotwords** — hint the engine with proper nouns (comma separated)
- **Low-confidence filter** — segments below an avg_logprob threshold (-0.8 / -0.6 / -0.4) are flagged in the manifest for review
- **Glossary** — under *Advanced options*, one `source=target` per line, injected into every translation request

## 🏗️ Architecture

```
app/
├── main.py            FastAPI: upload / start / cancel / WebSocket progress / download
├── settings.py        config.json read-write (API keys stay local)
├── model_store.py     On-demand model download (ModelScope/HF fallback + resume)
├── runtime_pack.py    Optional runtime loading (runtime/ → sys.path + DLL dirs)
├── pipeline/
│   ├── audio.py       ffmpeg audio extraction / subtitle burn-in
│   ├── vad.py         Speech screening + merging (core logic)
│   ├── asr.py         faster-whisper: batching, int8 auto-tune, hotwords, confidence
│   ├── qwen_asr.py    Qwen3-ASR optional engine (own implementation, Apache-2.0 weights)
│   ├── translate.py   LLM translation (context bridging / glossary / two-pass)
│   ├── srt.py         SRT writer with automatic line breaking
│   └── job.py         Orchestration: stage progress, rate-based ETA, cancellation
└── static/index.html  Single-page frontend: drag & drop, staged progress, preview
```

## 📦 Packaging

```bash
# Base build (~2.2GB, whisper engine only, ready out of the box)
.venv\Scripts\pyinstaller.exe Subtitles.spec --distpath dist

# Qwen runtime pack (optional, ~4.4GB: torch + transformers → runtime/ next to the exe)
python tools\make_runtime_pack.py
```

Two shapes: a 2.2GB base build that just works; users who want the Qwen engine unzip `runtime/` beside the executable and restart. Model weights are never bundled — they download from the in-app page.

> Native extensions in `runtime/` are cp313 and must match the exact Python minor version the exe was built with.

## 📄 License

Released under [MIT](LICENSE). This is an independent implementation containing no GPL code.
All dependencies are permissively licensed: faster-whisper (MIT), silero VAD (MIT), FastAPI (MIT),
ffmpeg (LGPL/GPL binaries distributed via static_ffmpeg). Whisper weights are MIT, Qwen3-ASR weights Apache-2.0.
See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

---

## 🇨🇳 中文说明

**把视频/音频变成高质量字幕 —— 本地 GPU 识别 + LLM 翻译，全程可视化。**

先初筛语音段（带时间戳），只对语音做识别，再翻译 —— 对话少的视频不再浪费算力扫全程。

### 主要特性

- 🎯 **智能语音初筛**：silero VAD 逐块扫描，安静低吟自动增益复检，只识别真正有人说话的片段
- ⚡ **批处理加速**：相邻语音块合并后批处理识别，长视频实测 **6.9x 实时**
- 🧠 **双引擎**：faster-whisper（默认，MIT）/ Qwen3-ASR（抗 BGM/噪声更强，模型页面按需下载）
- 🌐 **多种翻译大模型**：内置 DeepSeek / OpenAI GPT / Claude / Gemini / OpenRouter / Groq / Ollama 本地预设，一键切换；上下文桥接 + 术语表 + 可选两遍润色
- 📝 **可读性切分**：词级时间戳按标点/停顿切成 ≤6s 短句，拒绝 30 秒大长条
- 🖥️ **Web 界面**：拖拽上传、多阶段进度 + ETA 倒计时、字幕预览、中英双语切换
- 📦 **按需分发**：模型权重一律不打包，内置下载页支持断点续传/取消/删除

### 快速开始

```bash
git clone https://github.com/dklucifer/Subtitles.git
cd Subtitles
python -m venv .venv
.venv\Scripts\pip install -r requirements.txt
copy config.example.json config.json   # 填入你的翻译 API Key
run.bat
```

浏览器打开 <http://127.0.0.1:8765> 即可使用。系统要求 Windows 10/11 + Python 3.13+，GPU 为可选项（NVIDIA CUDA 更快，纯 CPU 也能跑）。密钥只保存在本机 `config.json`，不进代码库。

### 工作流程

```
视频 ─▶ ① 提取音频 ─▶ ② 语音初筛 ─▶ ③ 语音识别 ─▶ ④ LLM 翻译 ─▶ ⑤ 导出 SRT/烧录
```

- **① 提取音频**：内置 ffmpeg（static_ffmpeg，免安装）
- **② 语音初筛**：silero VAD 逐 10s 分块峰值归一化扫描，无语音块兜底低阈值二扫，输出覆盖率统计
- **③ 语音识别**：faster-whisper 只识别语音块，词级时间戳锚定原始时间轴；可选 Qwen3-ASR
- **④ 翻译**：上下文桥接（跨批次携带前文）、术语表强制译法、可选两遍翻译（草稿→润色）
- **⑤ 导出**：原文/译文/双语 SRT + 可选 ffmpeg libass 烧录

输出文件按语言自动命名，例如 `video.zh.srt`、`video.zh.bilingual.srt`、`video.orig.srt`。

### 许可

以 [MIT](LICENSE) 许可发布，为独立实现，不含任何 GPL 项目代码。依赖均为宽松许可组件（faster-whisper / silero VAD / FastAPI 均为 MIT，ffmpeg 为 LGPL/GPL 二进制）。详见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。
