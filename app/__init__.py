"""Subtitles — a consumer app that turns video/audio into bilingual subtitles.

Core pipeline (independent implementation, no third-party app code):
extract audio -> VAD speech screening (timestamps) -> merge blocks ->
ASR per block -> LLM translation -> SRT (bilingual).
"""
