"""Audio extraction via the bundled static ffmpeg (no system ffmpeg required)."""
import os
import subprocess

from static_ffmpeg import run as static_ffmpeg_run

_FFMPEG = None
# Windowless (PyInstaller --noconsole) parents make console children pop up a
# visible cmd window — suppress it on Windows.
_NO_WINDOW = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
_current = []   # running ffmpeg Popen(s), so shutdown can kill them


def ffmpeg_path():
    global _FFMPEG
    if _FFMPEG is None:
        _FFMPEG, _ = static_ffmpeg_run.get_or_fetch_platform_executables_else_raise()
    return _FFMPEG


def _run(cmd):
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            stdin=subprocess.DEVNULL, creationflags=_NO_WINDOW)
    _current.append(proc)
    try:
        _, err = proc.communicate()
    finally:
        if proc in _current:
            _current.remove(proc)
    if proc.returncode != 0:
        tail = (err or b"").decode(errors="replace").strip()[-400:]
        raise RuntimeError(f"ffmpeg failed (exit {proc.returncode}): {tail}")


def kill_current():
    """Terminate any running ffmpeg (called on server shutdown / cancel)."""
    for p in list(_current):
        try:
            p.terminate()
        except Exception:
            pass


def extract_audio(media_path, wav_path, sample_rate=16000):
    """Decode any media file to 16 kHz mono PCM WAV for VAD + ASR."""
    cmd = [ffmpeg_path(), "-y", "-v", "error", "-i", str(media_path),
           "-vn", "-ac", "1", "-ar", str(sample_rate), "-c:a", "pcm_s16le", str(wav_path)]
    _run(cmd)


def burn_subtitles(video_path, srt_path, out_path, font_size=24):
    """Optional: burn an SRT into the video with libass."""
    style = ("FontName=Microsoft YaHei,FontSize=%d,PrimaryColour=&H00FFFFFF,"
             "OutlineColour=&H00000000,BorderStyle=1,Outline=1.2,Shadow=0.6,MarginV=28" % font_size)
    cmd = [ffmpeg_path(), "-y", "-v", "error", "-i", str(video_path),
           "-vf", f"subtitles='{srt_path}':force_style='{style}'",
           "-c:a", "copy", str(out_path)]
    _run(cmd)
