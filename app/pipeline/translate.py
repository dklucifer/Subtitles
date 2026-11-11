"""Translation stage: any OpenAI-compatible endpoint (DeepSeek, Qwen, GPT, ...).

Accuracy features (all prompt-level, zero extra services):
- Context bridging: each batch carries the tail of the previous batch (source +
  translation) so tone, names and storyline stay continuous across batches.
- Glossary: user-defined "source=target" terms injected into every request.
- Two-pass mode (optional, premium): pass 1 literal draft, pass 2 polish with
  full context - roughly doubles API cost, hence off by default.
Batches lines, validates the response shape, retries once.
"""
import json


def parse_glossary(text):
    """Parse textarea input into [(src, dst)]. Accepts 'A=B', 'A->B', 'A=>B',
    one per line; lines without a separator are ignored."""
    pairs = []
    for line in (text or "").splitlines():
        line = line.strip()
        if not line:
            continue
        for sep in ("=>", "->", "="):
            if sep in line:
                src, dst = line.split(sep, 1)
                src, dst = src.strip(), dst.strip()
                if src and dst:
                    pairs.append((src, dst))
                break
    return pairs


def _glossary_block(glossary):
    if not glossary:
        return ""
    lines = "\n".join(f"- {s} => {d}" for s, d in glossary)
    return ("\nGlossary (mandatory translations, apply consistently):\n" + lines + "\n")


def translate_lines(lines, target_language, llm_cfg, glossary=None,
                    two_pass=False, on_progress=None, cancel=None):
    """lines: [{"index", "start", "end", "text"}] -> adds "translation" to each.
    Keeps numbering stable so context flows across batches."""
    from openai import OpenAI

    client = OpenAI(base_url=llm_cfg["base_url"], api_key=llm_cfg["api_key"], timeout=180)
    system = _system_prompt(target_language, glossary)

    batch_size = 40
    tail = []           # context bridge: last few (src, dst) of the previous batch
    results = {}
    n_batches = (len(lines) + batch_size - 1) // batch_size
    for bi in range(n_batches):
        if cancel is not None and cancel.is_set():
            raise _cancel()
        batch = lines[bi * batch_size:(bi + 1) * batch_size]
        payload = [{"index": ln["index"], "text": ln["text"]} for ln in batch]
        translation = _ask(client, system, payload, llm_cfg["model"], tail, cancel)
        for ln in batch:
            t = translation.get(str(ln["index"]))
            if not t:
                t = _find_by_index(translation, ln["index"]) or ln["text"]
            results[ln["index"]] = t
        tail = [(ln["text"], results[ln["index"]]) for ln in batch[-3:]]
        if on_progress:
            done_lines = min((bi + 1) * batch_size, len(lines))
            on_progress(done_lines if not two_pass else done_lines // 2, len(lines))
    for ln in lines:
        ln["translation"] = results.get(ln["index"], ln["text"])

    if two_pass:
        _polish(lines, target_language, llm_cfg, glossary, on_progress, cancel)
    return lines


def _polish(lines, target_language, llm_cfg, glossary, on_progress, cancel):
    """Second pass: refine the literal draft into natural, consistent subtitles
    with the same context-bridging scheme."""
    from openai import OpenAI

    client = OpenAI(base_url=llm_cfg["base_url"], api_key=llm_cfg["api_key"], timeout=180)
    system = _system_prompt(target_language, glossary) + (
        "\nThis is the POLISH pass. You receive the source text and a first-draft "
        "translation. Improve the draft: natural spoken register, consistent tone "
        "and terminology across the whole video, correct pronoun/reference resolution "
        "using the context lines, keep subtitle length reasonable. Never change the "
        "meaning. Same JSON output shape."
    )
    batch_size = 40
    tail = []
    n_batches = (len(lines) + batch_size - 1) // batch_size
    for bi in range(n_batches):
        if cancel is not None and cancel.is_set():
            raise _cancel()
        batch = lines[bi * batch_size:(bi + 1) * batch_size]
        payload = [{"index": ln["index"], "text": ln["text"],
                    "draft": ln.get("translation", "")} for ln in batch]
        translation = _ask(client, system, payload, llm_cfg["model"], tail, cancel)
        for ln in batch:
            t = translation.get(str(ln["index"]))
            if t:
                ln["translation"] = t
        tail = [(ln["text"], ln.get("translation", "")) for ln in batch[-3:]]
        if on_progress:
            done_lines = min((bi + 1) * batch_size, len(lines))
            on_progress(done_lines // 2 + done_lines // 2, len(lines))


def _system_prompt(target_language, glossary):
    return (
        "You are a professional subtitle translator. Translate each subtitle line into "
        f"{target_language}. Rules: keep the meaning natural and colloquial, like dubbing; "
        "one input line -> exactly one output line, same order, same indices; do not merge "
        "or split lines; keep names as-is unless an established translation exists. "
        + _glossary_block(glossary) +
        "Return ONLY a JSON object {\"lines\":[{\"index\":<int>,\"text\":\"<translation>\"}]}."
    )


def _context_block(tail):
    if not tail:
        return ""
    ctx = "\n".join(f"  {s} => {d}" for s, d in tail)
    return ("\nContext: the last lines already translated (for continuity of tone, "
            "characters and pronouns):\n" + ctx + "\n")


def _ask(client, system, payload, model, tail, cancel=None):
    """Attempt 0 sends response_format=json_object (DeepSeek/OpenAI/Gemini).
    Some OpenAI-compatible endpoints (e.g. Anthropic) reject that parameter, so
    attempt 1 retries without it and stresses the JSON shape in the message."""
    from openai import APIError
    user = _context_block(tail) + json.dumps({"lines": payload}, ensure_ascii=False)
    for attempt in (0, 1):
        try:
            kwargs = dict(
                model=model,
                messages=[{"role": "system", "content": system},
                          {"role": "user", "content": user}],
                temperature=0.3,
            )
            if attempt == 0:
                kwargs["response_format"] = {"type": "json_object"}
            else:
                user_retry = user + ('\nIMPORTANT: reply with ONLY a JSON object '
                                     '{"lines":[{"index":<int>,"text":"<translation>"}]}, '
                                     "no markdown, no extra text.")
                kwargs["messages"][1]["content"] = user_retry
            resp = client.chat.completions.create(**kwargs)
            data = json.loads(resp.choices[0].message.content or "{}")
            if isinstance(data.get("lines"), list):
                return _index_map(data["lines"])
        except (APIError, json.JSONDecodeError, KeyError, TypeError):
            if attempt == 1:
                return {}
    return {}


def _index_map(lines_list):
    out = {}
    for item in lines_list:
        try:
            out[str(int(item["index"]))] = str(item.get("text", "")).strip()
        except Exception:
            continue
    return out


def _find_by_index(translation, index):
    return translation.get(str(index)) or translation.get(index)


def _cancel():
    from app.pipeline.asr import CancelledError
    return CancelledError()
