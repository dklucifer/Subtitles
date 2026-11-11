<div align="center">

# Subtitles

**把视频/音频变成高质量双语字幕 — 本地 GPU 识别 + LLM 翻译，全程可视化**

[![Python](https://img.shields.io/badge/Python-3.13+-blue?logo=python&logoColor=white)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Platform](https://img.shields.io/badge/Platform-Windows-lightgrey?logo=windows)](#)

*先初筛语音段（带时间戳），只对语音做识别，再翻译 — 对话少的视频不再浪费算力扫全程*

</div>

---

## ✨ 特性

- 🎯 **智能语音初筛**：silero VAD 逐块扫描，安静低吟自动增益复检，只识别真正有人说话的片段
- ⚡ **批处理加速**：相邻语音块合并后批处理识别，长视频实测 **6.9x 实时**
- 🧠 **双引擎**：faster-whisper（默认，MIT）/ Qwen3-ASR（抗 BGM/噪声更强，模型页面按需下载）
- 🌐 **LLM 翻译**：OpenAI 兼容 API（DeepSeek/Qwen/GPT 均可），上下文桥接 + 术语表 + 可选两遍润色
- 📝 **可读性切分**：词级时间戳按标点/停顿切成 ≤6s 短句，拒绝 30 秒大长条
- 🖥️ **Web 界面**：拖拽上传、多阶段进度 + ETA 倒计时、字幕预览、中英双语切换
- 📦 **按需分发**：模型权重一律不打包，内置下载页支持断点续传/取消/删除

## 🚀 快速开始

### 环境要求

| 项目 | 要求 |
|---|---|
| 系统 | Windows 10/11（其他平台需自行适配 ffmpeg 路径） |
| Python | 3.13+ |
| GPU | NVIDIA CUDA（可选，CPU 也能跑但慢） |

### 安装运行

```bash
git clone https://github.com/<你的用户名>/Subtitles.git
cd Subtitles
python -m venv .venv
.venv\Scripts\pip install -r requirements.txt
copy config.example.json config.json   # 填入你的翻译 API Key
run.bat
```

浏览器打开 <http://127.0.0.1:8765> 即可使用。

> 🔑 密钥只保存在本机 `config.json`，不进代码库。

## ⚙️ 工作流程（5 阶段，全程可视化）

```
视频 ─▶ ① 提取音频 ─▶ ② 语音初筛 ─▶ ③ 语音识别 ─▶ ④ LLM 翻译 ─▶ ⑤ 导出 SRT/烧录
```

| 阶段 | 说明 |
|---|---|
| ① 提取音频 | 内置 ffmpeg（static_ffmpeg，免安装） |
| ② 语音初筛 | silero VAD 逐 10s 分块峰值归一化扫描，无语音块兜底低阈值二扫，输出覆盖率统计 |
| ③ 语音识别 | faster-whisper 只识别语音块，词级时间戳锚定原始时间轴；可选 Qwen3-ASR |
| ④ 翻译 | 上下文桥接（跨批次携带前文）、术语表强制译法、可选两遍翻译（草稿→润色） |
| ⑤ 导出 | 原文/译文/双语 SRT + 可选 ffmpeg libass 烧录 |

## 🧩 可选辅助功能

- **热词注入** — 告诉识别引擎专有名词（逗号分隔）
- **低置信度过滤** — avg_logprob 低于阈值（-0.8/-0.6/-0.4）的片段在 manifest 中标记，便于复核
- **术语表** — 界面「高级选项」，每行 `原文=译文`，注入全部翻译请求

## 🏗️ 架构

```
app/
├── main.py            FastAPI：上传/启动/取消/WebSocket 进度/下载
├── settings.py        config.json 读写（API key 仅存本机）
├── model_store.py     模型按需下载（ModelScope/HF 回退 + 断点续传）
├── runtime_pack.py    可选运行组件加载（runtime/ → sys.path + DLL 目录）
├── pipeline/
│   ├── audio.py       ffmpeg 抽音轨/烧录字幕
│   ├── vad.py         语音初筛 + 合并（核心逻辑）
│   ├── asr.py         faster-whisper：批处理 + int8 自适应 + 热词 + 置信度
│   ├── qwen_asr.py    Qwen3-ASR 可选引擎（自研实现，Apache-2.0 权重）
│   ├── translate.py   LLM 翻译（上下文桥接/术语表/两遍模式）
│   ├── srt.py         SRT 生成（自动断行）
│   └── job.py         任务编排：阶段进度、速率法 ETA 倒计时、可取消
└── static/index.html  单页前端：拖拽上传、多阶段进度、字幕预览
```

## 📦 打包分发

```bash
# 基础版（约 2.2GB，只含 whisper 引擎，开箱即用）
.venv\Scripts\pyinstaller.exe Subtitles.spec --distpath dist

# Qwen 运行组件（可选，约 4.4GB，torch + transformers，放 exe 同级 runtime/ 目录）
python tools\make_runtime_pack.py
```

两种形态：基础版 2.2GB 开箱即用；需要 Qwen 引擎的用户额外解压 `runtime/` 到程序目录，重启生效。模型权重一律不进包，界面内按需下载。

> runtime 里的原生扩展为 cp313，必须与 exe 使用同一 Python 小版本构建。

## 📄 许可说明

本项目以 [MIT](LICENSE) 许可发布，为独立实现，不含任何 GPL 项目的代码。
依赖均为宽松许可组件：faster-whisper（MIT）、silero VAD（MIT）、FastAPI（MIT）、
ffmpeg（LGPL/GPL 二进制随 static_ffmpeg 分发）。Whisper 模型权重 MIT，Qwen3-ASR 权重 Apache-2.0。
详见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。
