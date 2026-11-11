"""Audio extraction via the bundled static ffmpeg (no system ffmpeg required)."""
import subprocess

from static_ffmpeg import run as static_ffmpeg_run

_FFMPEG = None


def ffmpeg_path():
    global _FFMPEG
    if _FFMPEG is None:
        _FFMPEG, _ = static_ffmpeg_run.get_or_fetch_platform_executables_else_raise()
    return _FFMPEG


def extract_audio(media_path, wav_path, sample_rate=16000):
    """Decode any media file to 16 kHz mono PCM WAV for VAD + ASR."""
    cmd = [ffmpeg_path(), "-y", "-v", "error", "-i", str(media_path),
           "-vn", "-ac", "1", "-ar", str(sample_rate), "-c:a", "pcm_s16le", str(wav_path)]
    subprocess.run(cmd, check=True, capture_output=True)


def burn_subtitles(video_path, srt_path, out_path, font_size=24):
    """Optional: burn an SRT into the video with libass."""
    style = ("FontName=Microsoft YaHei,FontSize=%d,PrimaryColour=&H00FFFFFF,"
             "OutlineColour=&H00000000,BorderStyle=1,Outline=1.2,Shadow=0.6,MarginV=28" % font_size)
    cmd = [ffmpeg_path(), "-y", "-v", "error", "-i", str(video_path),
           "-vf", f"subtitles='{srt_path}':force_style='{style}'",
           "-c:a", "copy", str(out_path)]
    subprocess.run(cmd, check=True, capture_output=True)
