"""Job orchestration: runs the pipeline in a worker thread and emits progress events.

Every event carries: stage, stage percent, stage ETA (rate-based countdown), elapsed time
and an overall percent, so the UI can show live countdowns and a multi-stage timeline.
"""
import json
import os
import threading
import time
import uuid

from app.settings import ROOT, load
from app.pipeline import audio as audio_mod
from app.pipeline import vad as vad_mod
from app.pipeline import asr as asr_mod
from app.pipeline import srt as srt_mod
from app.pipeline.translate import translate_lines, parse_glossary
from app.pipeline.asr import CancelledError

STAGES = ["extract", "screen", "asr", "translate", "export"]
# rough weights for the overall bar
WEIGHTS = {"extract": 5, "screen": 5, "asr": 45, "translate": 40, "export": 5}
# naive speed factors used for ETA before a stage starts measuring (x realtime)
STAGE_SPEED = {"extract": 0.10, "screen": 0.15, "translate_batch_sec": 10.0}

# target-language display name -> ISO-style code for subtitle filenames
_LANG_CODES = [
    ("简体", "zh"), ("繁體", "zh-tw"), ("繁体", "zh-tw"),
    ("english", "en"), ("英", "en"),
    ("日本", "ja"), ("日语", "ja"), ("日文", "ja"),
    ("한국", "ko"), ("韩", "ko"),
    ("français", "fr"), ("法", "fr"),
    ("deutsch", "de"), ("德", "de"),
    ("español", "es"), ("西", "es"),
    ("русский", "ru"), ("俄", "ru"),
    ("português", "pt"), ("葡", "pt"),
    ("italiano", "it"), ("意", "it"),
]


def _lang_code(name):
    """Map a target-language label like 简体中文 / English to a filename tag (zh / en)."""
    import re as _re
    n = (name or "").strip().lower()
    for key, code in _LANG_CODES:
        if key.lower() in n:
            return code
    slug = _re.sub(r"[^a-z0-9-]", "", n)[:8]
    return slug or "sub"



class Job:
    def __init__(self, media_path, original_name, params, cfg):
        self.id = uuid.uuid4().hex[:8]
        self.media_path = media_path
        self.original_name = original_name
        self.params = params
        self.cfg = cfg
        self.cancel = threading.Event()
        self.thread = None
        self.listeners = []          # list[queue.Queue]
        self.state = {
            "id": self.id, "name": original_name, "status": "queued",
            "stage": None, "stage_pct": 0, "overall_pct": 0,
            "stage_eta": None, "elapsed": 0,
            "log": [], "segments": [], "files": [], "error": None,
            "info": {},               # duration, speech coverage, block count, device...
        }
        self._t0 = time.time()
        self._stage_t0 = None

    # ---- events ----
    def emit(self, event):
        for q in list(self.listeners):
            try:
                q.put_nowait(event)
            except Exception:
                pass

    def log(self, text):
        self.state["log"].append(text)
        if len(self.state["log"]) > 400:
            self.state["log"] = self.state["log"][-400:]
        self.emit({"type": "log", "text": text})

    def stage(self, name, pct, eta=None, msg=""):
        self.state["stage"] = name
        self.state["stage_pct"] = int(pct)
        self.state["stage_eta"] = eta
        self.state["elapsed"] = round(time.time() - self._t0, 1)
        done_weight = sum(WEIGHTS[s] for s in STAGES[:STAGES.index(name)])
        self.state["overall_pct"] = min(99, done_weight + int(WEIGHTS[name] * pct / 100))
        self.emit({"type": "progress", "stage": name, "pct": int(pct),
                   "eta": eta, "elapsed": self.state["elapsed"],
                   "overall": self.state["overall_pct"], "msg": msg})

    def stage_done(self, name):
        idx = STAGES.index(name)
        self.state["overall_pct"] = sum(WEIGHTS[s] for s in STAGES[:idx + 1])

    # ---- pipeline ----
    def run(self):
        try:
            self._run()
        except CancelledError:
            self.state["status"] = "cancelled"
            self.emit({"type": "cancelled"})
        except Exception as e:
            self.state["status"] = "error"
            self.state["error"] = str(e)
            self.emit({"type": "error", "text": str(e)})

    def _run(self):
        self.state["status"] = "running"
        p, cfg = self.params, self.cfg
        out_dir = os.path.join(ROOT, "output", self.id)
        os.makedirs(out_dir, exist_ok=True)
        self.state["info"]["out_dir"] = out_dir
        wav = os.path.join(out_dir, "audio.wav")

        # ---- stage 1: extract ----
        self._stage_start("extract")
        self.log("Extracting audio...")
        audio_mod.extract_audio(self.media_path, wav)
        _, duration = vad_mod.load_wav(wav)
        self.state["info"]["duration"] = round(duration, 1)
        self.state["info"]["media"] = self.original_name
        self.stage("extract", 100, 0)
        self.stage_done("extract")

        # ---- stage 2: speech screening (the core logic) ----
        self._stage_start("screen")
        self.log("Screening speech regions (VAD)...")
        blocks, duration, coverage = vad_mod.screen_speech(
            wav, max_gap=p.get("min_block_gap", 0.5), min_len=p.get("min_block_len", 0.6))
        if not blocks:
            raise RuntimeError("No speech detected in this media file.")
        speech_sec = sum(e - s for s, e in blocks)
        self.state["info"].update({
            "blocks": len(blocks), "coverage": round(coverage * 100, 1),
            "speech_sec": round(speech_sec, 1),
        })
        self.log(f"Found {len(blocks)} speech block(s); speech {speech_sec:.0f}s / "
                 f"{duration:.0f}s ({coverage * 100:.0f}%) — ASR will skip the rest.")
        self.stage("screen", 100, 0, f"{len(blocks)} blocks")
        self.stage_done("screen")

        # ---- stage 3: ASR on speech blocks only ----
        self._stage_start("asr")
        asr_start = time.time()

        asr_logged_pct = [0.0]

        def on_asr_progress(done, total, marks=[]):
            # Progress arrives in bursts (one batch of clips finishes at once), so
            # interval-based estimates swing wildly. Use the average rate since the
            # stage started instead: it is stable and converges within seconds.
            if done < 3:
                eta = None                      # model warmup, no honest number yet
            else:
                rate = done / max(0.001, time.time() - asr_start)
                eta = max(0, (total - done) / rate)
            pct = done * 100.0 / total
            self.stage("asr", pct,
                       round(eta) if eta is not None else None,
                       f"{done}/{total} blocks")
            # keep the run log alive during this long stage (every 10%)
            if done < total and pct - asr_logged_pct[0] >= 10:
                asr_logged_pct[0] = pct
                if eta is not None:
                    self.log(f"ASR {int(pct)}% · {done}/{total} blocks · ETA {int(eta)}s")

        engine = p.get("asr_engine") or cfg["asr"].get("engine", "whisper")
        hotwords = (p.get("hotwords") or "").strip() or None
        min_conf = float(p.get("min_confidence") or 0.0)
        language = None if p.get("src_language", "auto") == "auto" else p["src_language"]
        if engine.startswith("qwen"):
            from app.pipeline import qwen_asr
            if not qwen_asr.is_ready(engine):
                raise RuntimeError(
                    f"Qwen3-ASR engine '{engine}' is not ready: model weights are "
                    "missing under models/ (open /download to fetch them) or "
                    "torch/transformers<5.13 is unavailable. "
                    "Select the whisper engine instead.")
            segments, device = qwen_asr.transcribe_blocks(
                wav, blocks, engine=engine,
                device=p.get("device") or cfg["asr"]["device"],
                language=language, hotwords=hotwords,
                on_progress=on_asr_progress, cancel=self.cancel)
        else:
            segments, device = asr_mod.transcribe_blocks(
                wav, blocks, model_name=p.get("asr_model") or cfg["asr"]["model"],
                device=p.get("device") or cfg["asr"]["device"],
                compute=cfg["asr"].get("compute", "auto"),
                batch=cfg["asr"].get("batch", True),
                language=language,
                hotwords=hotwords, min_confidence=min_conf,
                on_progress=on_asr_progress, cancel=self.cancel)
        self.state["info"]["device"] = device
        self.state["info"]["engine"] = engine
        self.state["info"]["segments"] = len(segments)
        low = sum(1 for s in segments if s.get("low_confidence"))
        if low:
            self.log(f"{low} segment(s) flagged low-confidence (see manifest).")
        asr_elapsed = time.time() - asr_start
        self.log(f"ASR done: {len(segments)} segments in {asr_elapsed:.0f}s ({device}, engine={engine}).")
        if engine == "whisper" and asr_mod.LAST_MODE.get("error"):
            self.log(f"Batched decoding fell back to per-block: {asr_mod.LAST_MODE['error']}")
        self.stage("asr", 100, 0)
        self.stage_done("asr")

        # ---- stage 4: translate ----
        api_key = cfg["llm"].get("api_key") or ""
        lines = [{"index": i, "start": s["start"], "end": s["end"], "text": s["text"]}
                 for i, s in enumerate(segments)]
        if api_key:
            self._stage_start("translate")
            tr_start = time.time()

            tr_logged_pct = [0.0]

            def on_tr_progress(done, total):
                elapsed = time.time() - tr_start
                per = elapsed / max(done, 1)
                eta = max(0, (total - done) * per)
                pct = done * 100.0 / total
                self.stage("translate", pct, round(eta), f"{done}/{total} lines")
                # keep the run log alive during this long stage (every 20%)
                if done < total and pct - tr_logged_pct[0] >= 20:
                    tr_logged_pct[0] = pct
                    self.log(f"翻译中 {int(pct)}% · {done}/{total} 行 · ETA {int(eta)}s")

            self.log(f"Translating {len(lines)} lines -> {p.get('target_language')}...")
            translate_lines(lines, p.get("target_language") or cfg["subtitle"]["target_language"],
                            cfg["llm"],
                            glossary=parse_glossary(p.get("glossary") or cfg["subtitle"].get("glossary", "")),
                            two_pass=bool(p.get("two_pass") or cfg["subtitle"].get("two_pass", False)),
                            on_progress=on_tr_progress, cancel=self.cancel)
            for i, ln in enumerate(lines):
                segments[i]["translation"] = ln["translation"]
            self.stage("translate", 100, 0)
            self.stage_done("translate")
        else:
            self.log("No LLM API key configured — skipped translation (source SRT only).")
            for s in segments:
                s["translation"] = ""

        # ---- stage 5: export ----
        self._stage_start("export")

        # subtitle filenames carry a language tag: 原名.zh.srt / 原名.zh.bilingual.srt
        base = os.path.splitext(self.original_name)[0]
        tgt_code = _lang_code(p.get("target_language") or cfg["subtitle"]["target_language"])

        self.stage("export", 30, 2)
        src_path = srt_mod.write_srt(segments, os.path.join(out_dir, f"{base}.orig.srt"),
                                     mode="source", max_line_chars=cfg["subtitle"]["max_line_chars"])
        files = [{"name": f"{base}.orig.srt", "path": src_path}]
        if any(s.get("translation") for s in segments):
            tgt_path = srt_mod.write_srt(segments, os.path.join(out_dir, f"{base}.{tgt_code}.srt"),
                                         mode="translation", max_line_chars=cfg["subtitle"]["max_line_chars"])
            bi_path = srt_mod.write_srt(segments, os.path.join(out_dir, f"{base}.{tgt_code}.bilingual.srt"),
                                        mode="bilingual", max_line_chars=cfg["subtitle"]["max_line_chars"])
            files += [{"name": f"{base}.{tgt_code}.srt", "path": tgt_path},
                      {"name": f"{base}.{tgt_code}.bilingual.srt", "path": bi_path}]
        if p.get("burn") and self.original_name.lower().endswith((".mp4", ".mkv", ".mov", ".webm")):
            srt_for_burn = bi_path if any(s.get("translation") for s in segments) else src_path
            self.stage("export", 60, None)
            self.log("Burning subtitles into video...")
            out_video = os.path.join(out_dir, "video.subtitled" + os.path.splitext(self.original_name)[1])
            audio_mod.burn_subtitles(self.media_path, srt_for_burn.replace("\\", "/").replace(":", "\\:"),
                                     out_video, font_size=int(p.get("font_size", 24)))
            files.append({"name": os.path.basename(out_video), "path": out_video})
        self.state["files"] = [{"name": f["name"], "path": f["path"]} for f in files]
        with open(os.path.join(out_dir, "manifest.json"), "w", encoding="utf-8") as f:
            json.dump({"segments": segments, "files": files, "info": self.state["info"]},
                      f, ensure_ascii=False)
        self.state["segments"] = [
            {"start": s["start"], "end": s["end"], "text": s["text"], "translation": s.get("translation", "")}
            for s in segments[:200]]
        self.stage("export", 100, 0)
        self.state["status"] = "done"
        self.emit({"type": "done", "files": self.state["files"]})

    def _stage_start(self, name):
        self._stage_t0 = time.time()
        est = self._stage_estimate(name)
        self.stage(name, 0, est, "starting")

    def _stage_estimate(self, name):
        info = self.state["info"]
        if name == "extract":
            return None
        if name == "screen":
            return None
        if name == "asr":
            sp = info.get("speech_sec", 0)
            engine = self.params.get("asr_engine") or self.cfg["asr"].get("engine", "whisper")
            if engine.startswith("qwen"):
                factor = 0.5 if "0.6b" in engine else 0.8
            else:
                gpu = (self.params.get("device") or self.cfg["asr"]["device"]) != "cpu"
                factor = 0.03 if gpu else 0.35
            return round(sp * factor)
        if name == "translate":
            return None
        return None


class JobManager:
    def __init__(self):
        self.job = None
        self._lock = threading.Lock()

    def start(self, media_path, original_name, params, cfg):
        with self._lock:
            if self.job and self.job.state["status"] in ("queued", "running"):
                raise RuntimeError("A job is already running. Cancel it first.")
            job = Job(media_path, original_name, params, cfg)
            self.job = job
            job.thread = threading.Thread(target=job.run, daemon=True)
            job.thread.start()
            return job

    def snapshot(self):
        return dict(self.job.state) if self.job else None


MANAGER = JobManager()
