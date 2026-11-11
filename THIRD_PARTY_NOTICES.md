# 第三方组件与许可证说明 / Third-party notices

Subtitles 本体代码为自有实现，**不包含任何 GPL/AGPL 代码**。

| 组件 | 许可证 | 用途 | 说明 |
|---|---|---|---|
| faster-whisper | MIT | 默认 ASR 引擎 | 允许闭源商用 |
| CTranslate2 | MIT | 推理后端 | |
| Silero VAD（通过 faster-whisper 内置 onnx） | MIT | 语音初筛 | |
| static-ffmpeg / FFmpeg | LGPL/GPL 双许可（本程序仅调用 ffmpeg 可执行文件，不链接其库） | 音频抽取、字幕烧录 | 以独立进程调用，无传染 |
| FastAPI / uvicorn | MIT / BSD | Web 服务 | |
| openai-python | Apache-2.0 | 调用翻译大模型 API | |
| PyAV / av | BSD-3 | 音频解码 | |
| Qwen3-ASR-0.6B-hf / Qwen3-ASR-1.7B-hf（模型权重） | **Apache-2.0** | 可选 ASR 引擎 | 可商用，需保留版权与许可声明 |
| transformers（Hugging Face） | Apache-2.0 | Qwen3-ASR 官方推理 API | 仅使用官方公开 API |
| PyTorch | BSD-3 | Qwen3-ASR 运行时 | |
| accelerate | Apache-2.0 | 模型加载 | |

## GPL 边界声明

- `app/pipeline/qwen_asr.py` 为**独立编写**的实现，只依赖：
  1. Apache-2.0 的模型权重（Qwen3-ASR-*-hf）；
  2. Apache-2.0 的 transformers 公开 API（`AutoProcessor` / `AutoModelForMultimodalLM` / `apply_transcription_request`）。
- 未复制、未改编、未链接任何 GPL 项目（包括 VideoLingo）的源代码，也未读取其实现细节作为依据。
- 因此 Subtitles 可以闭源分发或商业授权，不受 GPL 传染。

## 可选运行组件（runtime/）

Qwen 引擎所需的 PyTorch / transformers 等依赖**不随主程序分发**，而是作为可选的
`runtime/` 组件目录由用户自行放置。若你对外分发该组件目录，请一并附上：

- PyTorch：BSD-3-Clause（含其内置的 CUDA/cuDNN 运行库，受 NVIDIA 许可约束）
- transformers：Apache-2.0
- accelerate：Apache-2.0
- soundfile：BSD-3（含 libsndfile，LGPL-2.1）
- 其余传递依赖（huggingface_hub、tokenizers、safetensors、sympy、networkx 等）：Apache-2.0 / MIT / BSD

这些组件与主程序之间只是 `sys.path` 层面的加载关系，不改变 Subtitles 本体代码的许可。

## 模型许可文本

Qwen3-ASR 权重采用 Apache License 2.0，分发时请随附：
- Apache License 2.0 全文（https://www.apache.org/licenses/LICENSE-2.0）
- Qwen3-ASR 模型卡中的版权声明（© Alibaba Cloud / Qwen Team）
