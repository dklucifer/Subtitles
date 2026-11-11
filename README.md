# Subtitles

C 端字幕工具：把视频/音频变成双语字幕。核心逻辑：**先初筛语音段（带时间戳），只对语音做识别，再翻译**——
对话少的视频不再浪费算力扫全程。

## 快速开始

```
run.bat            （或 .venv\Scripts\python.exe -m app.main）
浏览器打开 http://127.0.0.1:8765
```

首次使用在「翻译 API 设置」里填 OpenAI 兼容端点（DeepSeek / Qwen / GPT 均可），
密钥只保存在本机 `config.json`，不进代码库。

## 流程（5 阶段，全程可视化 + 倒计时）

1. **提取音频** — 内置 ffmpeg（static_ffmpeg，免安装）
2. **语音初筛** — silero VAD（onnxruntime），逐 10s 分块峰值归一化扫描：
   安静低吟先"调大音量"再检测；无语音块兜底低阈值二扫；间隔 ≤0.5s 的语音段合并，
   <0.6s 丢弃；输出覆盖率统计（如"语音 17s / 22s，77%"）
3. **语音识别** — 默认 faster-whisper（MIT），**只识别语音块**，词级时间戳锚定原始时间轴；
   相邻语音块合并成 ≤27s 的 clip 后批处理（实测 6.9x 实时），超长对白块先在"最安静处"切段，
   再按标点/停顿切成 ≤6s 的可读字幕。可选 Qwen3-ASR 引擎（Apache-2.0 权重，抗 BGM/噪声更强，
   0.6B / 1.7B 两档，**模型可在 /download 页按需下载，支持断点续传**；运行需额外的
   `runtime/` 组件目录，见下文「打包」）
4. **翻译** — OpenAI 兼容 API 分批翻译，JSON 结构校验 + 失败重试；
   **上下文桥接**（跨批次携带前文，人名/语气连贯）、**术语表**强制译法、
   可选**两遍翻译**（草稿 → 润色，高级模式）
5. **导出** — 原文 / 译文 / 双语 SRT + 可选 ffmpeg libass 烧录

## 可选辅助功能

- **热词注入**：告诉识别引擎专有名词（逗号分隔）
- **低置信度过滤**：avg_logprob 低于阈值（-0.8/-0.6/-0.4）的片段在 manifest 中标记，便于复核
- **术语表**：界面「高级选项」，每行 `原文=译文`，注入全部翻译请求

## 架构

```
app/
├── main.py            FastAPI：上传 / 启动 / 取消 / WebSocket 进度 / 下载
├── settings.py        config.json 读写（API key 仅存本机）
├── pipeline/
│   ├── audio.py       ffmpeg 抽音轨 / 烧录字幕
│   ├── vad.py         语音初筛 + 合并（核心逻辑）
│   ├── asr.py         faster-whisper：批处理 + int8 自适应 + 热词 + 置信度
│   ├── qwen_asr.py    Qwen3-ASR 可选引擎（自研实现，Apache-2.0 权重）
├── model_store.py     模型按需下载（ModelScope/HF 回退 + 断点续传）
├── runtime_pack.py    可选运行组件加载（<程序目录>/runtime → sys.path + DLL 目录）
│   ├── translate.py   LLM 翻译（上下文桥接 / 术语表 / 两遍模式）
│   ├── srt.py         SRT 生成（自动断行）
│   └── job.py         任务编排：阶段进度、速率法 ETA 倒计时、可取消
└── static/index.html  单页前端：中英双语切换、拖拽上传、多阶段进度、字幕预览
```

## 打包

```
基础版（默认，约 2.2GB）：.venv\Scripts\pyinstaller.exe Subtitles.spec --distpath dist
Qwen 运行组件（可选，约 4.4GB）：python tools\make_runtime_pack.py
```

**两种分发形态**

- 基础版：只含 whisper 引擎，开箱即用，2.2GB
- Qwen 引擎需要额外的运行组件（torch + transformers）。它**不在 exe 里**，而是作为
  一个可选目录 `runtime/` 放在 exe 同级，启动时自动加载（`app/runtime_pack.py`）。
  这样不用 Qwen 的用户只需要下载 2.2GB；需要的人再加 4.4GB。

`tools\make_runtime_pack.py` 从项目 venv 生成该目录（已裁掉 torch 的 `.lib` 链接库等，
省 2.7GB）。生成后把 `runtime/` 与 `Subtitles.exe` 放同级即可，也可以压成
`runtime-pack.zip` 单独分发，用户解压到程序目录后重启生效。

模型权重一律不进包：whisper 首次运行自动下载；Qwen3-ASR 在界面选该引擎时会提示，
点「去下载」进入 `/download` 页（支持断点续传、取消、删除）。

> 运行组件里带的是 cp313 的原生扩展，必须与 exe 使用同一个 Python 小版本构建。

## 许可说明

本程序为独立实现，不含任何 GPL 项目的代码。依赖均为 MIT/Apache 许可组件：
faster-whisper、silero VAD（经 faster-whisper 内置 onnx 模型）、FastAPI、ffmpeg（LGPL/GPL 二进制随 static_ffmpeg 分发）。
Whisper 模型权重 MIT。
