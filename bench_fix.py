"""Benchmark: slice-based transcribe on the long video's audio (the bug fix)."""
import sys, time
sys.path.insert(0, r"C:\Subtitles")
import os
os.chdir(r"C:\Subtitles")

from app.pipeline.audio import extract_audio, ffmpeg_path
print("ffmpeg:", ffmpeg_path())

t0 = time.time()
wav = r"C:\Subtitles\work\bench_long.wav"
extract_audio(r"C:\Subtitles\work\4024_DVMM-368.mp4", wav)
print("extract: %.1fs" % (time.time() - t0))

import numpy as np, wave
with wave.open(wav, "rb") as w:
    sr = w.getframerate()
    pcm = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
audio = pcm.astype(np.float32) / 32768.0
print("audio: %.0fs @ %dHz" % (len(audio)/sr, sr))

from app.pipeline.asr import transcribe_blocks
# 20 synthetic 30s blocks spread across the track (speed test, not quality)
dur = len(audio) / sr
blocks = [(i * dur / 25, i * dur / 25 + 30) for i in range(2, 22)]

t0 = time.time()
segs, dev = transcribe_blocks(
    wav, blocks, model_name="small", device="auto", language=None,
    on_progress=lambda d, t: print("  %d/%d  %.1fs elapsed" % (d, t, time.time() - t0)))
total = time.time() - t0
print("DEVICE:", dev)
print("20 blocks (600s speech) in %.1fs -> %.2fs per 30s block, %.0fx realtime" % (
    total, total / 20, 600 / total))
est_total = 2360 / (600 / total)  # 2360s speech in this video
print("projected ASR for 2360s speech: %.0f min" % (est_total / 60))
