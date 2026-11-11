"""FastAPI server: upload, start, live progress over WebSocket, download results."""
import asyncio
import os
import queue
import threading

from fastapi import FastAPI, UploadFile, File, Form, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from app.settings import ROOT, BUNDLE_DIR, load, save
from app.pipeline.job import MANAGER

# Optional Qwen runtime pack (torch / transformers) living in <app>/runtime.
# Must run before anything imports torch, hence at module import time.
from app.runtime_pack import activate as _activate_runtime
_activate_runtime()

app = FastAPI(title="Subtitles")
WORK_DIR = os.path.join(ROOT, "work")
os.makedirs(WORK_DIR, exist_ok=True)
STATIC_DIR = os.path.join(BUNDLE_DIR, "app", "static")

# Windowless mode (pythonw / pyinstaller --noconsole) leaves stdout/stderr as
# None; uvicorn's logging would choke on that, so point them at devnull.
import sys

if sys.stdout is None:
    sys.stdout = open(os.devnull, "w", encoding="utf-8")
if sys.stderr is None:
    sys.stderr = open(os.devnull, "w", encoding="utf-8")


@app.get("/")
def index():
    # no-cache so UI updates show up on plain refresh, not just Ctrl+F5
    return FileResponse(os.path.join(STATIC_DIR, "index.html"),
                        headers={"Cache-Control": "no-cache, no-store, must-revalidate"})


@app.get("/api/config")
def get_config():
    cfg = load()
    masked = dict(cfg)
    masked["llm"] = {k: ("•" * 8 if k == "api_key" and v else v) for k, v in cfg["llm"].items()}
    masked["has_api_key"] = bool(cfg["llm"].get("api_key"))
    try:
        from app.pipeline import qwen_asr
        masked["qwen_available"] = qwen_asr.available()
    except Exception:
        masked["qwen_available"] = False
    return masked


# ------------------------------------------------------------ model store
@app.get("/download")
def download_page():
    return FileResponse(os.path.join(STATIC_DIR, "download.html"),
                        headers={"Cache-Control": "no-cache, no-store, must-revalidate"})


@app.get("/api/models")
def models_status():
    from app import model_store
    return model_store.status()


@app.post("/api/models/download")
def models_download(body: dict):
    from app import model_store
    return model_store.start(body.get("engine") or "")


@app.post("/api/models/cancel")
def models_cancel():
    from app import model_store
    return model_store.cancel()


@app.post("/api/models/delete")
def models_delete(body: dict):
    from app import model_store
    return model_store.delete(body.get("engine") or "")


@app.get("/api/runtime_check")
def runtime_check():
    """Diagnose the optional runtime pack: what imported, what failed, and why."""
    import sys
    import traceback

    from app import runtime_pack
    out = {"frozen": bool(getattr(sys, "frozen", False)),
           "runtime": runtime_pack.status(),
           "python": sys.version.split()[0],
           "path_tail": sys.path[-3:]}
    try:
        import torch
        out["torch"] = torch.__version__
        out["cuda"] = bool(torch.cuda.is_available())
    except Exception:
        out["torch_error"] = traceback.format_exc()[-1200:]
    try:
        import transformers
        out["transformers"] = transformers.__version__
    except Exception:
        out["transformers_error"] = traceback.format_exc()[-1200:]
    for _name in ("AutoProcessor", "AutoModelForMultimodalLM"):
        try:
            getattr(__import__("transformers"), _name)
        except Exception:
            out[f"{_name}_error"] = traceback.format_exc()[-2500:]
    return out


@app.post("/api/config")
def set_config(cfg: dict):
    from app.settings import _deep_merge
    current = load()
    llm = cfg.get("llm", {})
    # an empty/masked key keeps the stored one
    if not llm.get("api_key") or set(llm["api_key"]) == {"•"}:
        llm["api_key"] = current["llm"]["api_key"]
    current = _deep_merge(current, {k: v for k, v in cfg.items() if k != "llm"})
    current["llm"] = llm
    save(current)
    return {"ok": True}


@app.post("/api/test_llm")
def test_llm():
    """Send a tiny chat completion so the user can verify the API works."""
    import time as _time

    cfg = load()["llm"]
    if not cfg.get("base_url") or not cfg.get("api_key") or not cfg.get("model"):
        return {"ok": False, "error": "incomplete config"}
    from openai import OpenAI

    t0 = _time.time()
    try:
        client = OpenAI(base_url=cfg["base_url"], api_key=cfg["api_key"], timeout=15)
        resp = client.chat.completions.create(
            model=cfg["model"],
            messages=[{"role": "user", "content": "ping, reply with the single word: pong"}],
            max_tokens=5,
        )
        ms = int((_time.time() - t0) * 1000)
        return {"ok": True, "ms": ms, "model": cfg["model"],
                "reply": (resp.choices[0].message.content or "").strip()[:20]}
    except Exception as e:
        msg = str(e)
        return {"ok": False, "ms": int((_time.time() - t0) * 1000),
                "error": msg[:300] if msg else e.__class__.__name__}


@app.post("/api/start")
async def start_job(
    file: UploadFile = File(...),
    target_language: str = Form("简体中文"),
    src_language: str = Form("auto"),
    bilingual: str = Form("true"),
    burn: str = Form("false"),
    asr_model: str = Form("small"),
    asr_engine: str = Form("whisper"),
    device: str = Form("auto"),
    font_size: int = Form(24),
    min_block_gap: float = Form(0.5),
    min_block_len: float = Form(0.6),
    hotwords: str = Form(""),
    min_confidence: float = Form(0.0),
    glossary: str = Form(""),
    two_pass: str = Form("false"),
):
    # Fail fast *before* uploading a 3 GB movie: the engine's weights may be absent.
    if asr_engine.startswith("qwen"):
        from app import model_store
        from app.pipeline import qwen_asr
        if not qwen_asr.is_ready(asr_engine):
            reason = "model_missing" if model_store.deps_ok() else "runtime_missing"
            return JSONResponse(
                {"error": reason, "code": reason, "engine": asr_engine,
                 "download_url": f"/download?engine={asr_engine}"},
                status_code=409)
    save_path = os.path.join(WORK_DIR, f"{os.getpid()}_{file.filename}")
    with open(save_path, "wb") as f:
        while chunk := await file.read(1 << 20):
            f.write(chunk)
    params = {
        "target_language": target_language,
        "src_language": src_language,
        "bilingual": bilingual == "true",
        "burn": burn == "true",
        "asr_model": asr_model,
        "asr_engine": asr_engine,
        "device": device,
        "font_size": int(font_size),
        "min_block_gap": min_block_gap,
        "min_block_len": min_block_len,
        "hotwords": hotwords,
        "min_confidence": float(min_confidence),
        "glossary": glossary,
        "two_pass": two_pass == "true",
    }
    try:
        job = MANAGER.start(save_path, file.filename, params, load())
    except RuntimeError as e:
        return JSONResponse({"error": str(e)}, status_code=409)
    return {"ok": True, "job_id": job.id}


@app.post("/api/cancel")
def cancel_job():
    if MANAGER.job:
        MANAGER.job.cancel.set()
    return {"ok": True}


@app.post("/api/shutdown")
def shutdown_server():
    """Clean exit for stop.bat - the windowless app has no close button."""
    threading.Timer(0.5, os._exit, (0,)).start()
    return {"ok": True}


@app.get("/api/state")
def state():
    snap = MANAGER.snapshot()
    if snap:
        snap["log"] = snap["log"][-30:]
        snap["segments"] = snap["segments"][:30]
    return snap or {"status": "idle"}


@app.get("/api/download/{name}")
def download(name: str):
    if MANAGER.job:
        for f in MANAGER.job.state["files"]:
            if f["name"] == name:
                return FileResponse(f["path"], filename=name)
    return JSONResponse({"error": "not found"}, status_code=404)


@app.post("/api/open_folder")
def open_folder():
    """Open the job's output directory in Windows Explorer."""
    job = MANAGER.job
    out_dir = (job.state.get("info") or {}).get("out_dir") if job else None
    if not out_dir or not os.path.isdir(out_dir):
        return JSONResponse({"error": "no output directory yet"}, status_code=404)
    try:
        os.startfile(out_dir)   # Windows Explorer; local trusted app
        return {"ok": True}
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@app.websocket("/ws")
async def ws_progress(ws: WebSocket):
    await ws.accept()
    q = queue.Queue()
    job = MANAGER.job
    if job:
        job.listeners.append(q)
        await ws.send_json({"type": "snapshot", "state": {**job.state, "log": job.state["log"][-30:]}})
    else:
        await ws.send_json({"type": "snapshot", "state": {"status": "idle"}})
    try:
        while True:
            drained = False
            while True:
                try:
                    await ws.send_json(q.get_nowait())
                    drained = True
                except queue.Empty:
                    break
            if not drained:
                # still alive; light heartbeat lets the client keep its countdown ticking
                await asyncio.sleep(0.4)
            else:
                await asyncio.sleep(0.05)
    except WebSocketDisconnect:
        pass
    finally:
        if job:
            try:
                job.listeners.remove(q)
            except ValueError:
                pass


app.mount("/", StaticFiles(directory=STATIC_DIR), name="static")


def serve():
    import socket
    import threading
    import time
    import webbrowser

    import uvicorn

    cfg = load()
    port = cfg["server"]["port"]
    url = f"http://127.0.0.1:{port}"

    # Already running? Don't start a second instance (it would die on the
    # busy port) - just bring up the existing app in the browser and exit.
    try:
        s = socket.create_connection(("127.0.0.1", port), timeout=0.5)
        s.close()
        webbrowser.open(url)
        print(f"Subtitles is already running at {url}")
        return
    except OSError:
        pass

    def open_browser():
        time.sleep(1.5)  # give uvicorn a moment to bind
        webbrowser.open(url)

    threading.Timer(0, open_browser).start()
    uvicorn.run(app, host="127.0.0.1", port=port, log_level="warning")


if __name__ == "__main__":
    serve()
