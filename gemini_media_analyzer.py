#!/usr/bin/env python3
"""gemini_media_analyzer.py — objective Gemini media transcription/observation CLI.

This tool is deliberately dumb: it transcribes and describes media, but does not
fact-check, classify propaganda, infer intent, or decide truth.

Env: GOOGLE_API_KEY (required); GEMINI_API_KEY is accepted as a fallback.
"""

from __future__ import annotations

import argparse
import json
import mimetypes
import os
import pathlib
import sys
import time
from typing import Any

_DEFAULT_MODEL = "gemini-3.5-flash"
_DEFAULT_TIMEOUT_SECONDS = 300
_POLL_INTERVAL_SECONDS = 2
_SUPPORTED_MEDIA_PREFIXES = ("audio/", "video/", "image/")
_SUPPORTED_EXTRA_MIME_TYPES = {"application/pdf"}

_OBJECTIVE_SYSTEM_INSTRUCTION = """You are a media transcription and observation engine.
Your job is to report only what is present in the supplied media.
Do not fact-check claims.
Do not decide whether claims are true or false.
Do not classify propaganda, misinformation, persuasion, ideology, or bias.
Do not infer intent, motive, or hidden meaning.
Do not add external context or web research.
Describe observable audio, speech, visible text, and visual events only.
If something is uncertain, mark it as uncertain.
Return valid JSON only, with no markdown fences.
"""

_BASE_SCHEMA = {
    "media": {
        "path": "string",
        "mime_type": "string",
        "duration_seconds": "number|null if unknown",
    },
    "audio": {
        "detected_languages": ["string"],
        "transcript": [
            {
                "start": "HH:MM:SS or null",
                "end": "HH:MM:SS or null",
                "text": "verbatim speech",
                "translation_en": "English translation or null",
                "confidence": "high|medium|low|unknown",
            }
        ],
        "unclear_segments": [
            {
                "timestamp": "HH:MM:SS or range",
                "reason": "inaudible|overlap|music|other",
                "best_effort_text": "string|null",
            }
        ],
    },
    "visual": {
        "onscreen_text": [
            {
                "timestamp": "HH:MM:SS or null",
                "text": "visible text",
                "confidence": "high|medium|low|unknown",
            }
        ],
        "scene_observations": [
            {
                "timestamp": "HH:MM:SS or null",
                "description": "objective description of visible scene/event",
            }
        ],
    },
    "model_notes": {
        "limitations": ["string"],
        "safety_blocks": ["string"],
    },
}


def get_api_key() -> str:
    """Read Gemini API key from environment or exit."""
    key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
    if not key:
        print(
            "ERROR: GOOGLE_API_KEY environment variable not set "
            "(GEMINI_API_KEY fallback also empty)",
            file=sys.stderr,
        )
        sys.exit(1)
    return key


def make_client(key: str):
    """Build a google-genai client."""
    from google import genai

    return genai.Client(api_key=key)


def detect_mime(path: pathlib.Path) -> str:
    """Return best-guess MIME type for a file."""
    mime, _ = mimetypes.guess_type(str(path))
    return mime or "application/octet-stream"


def validate_media_path(file_path: str) -> pathlib.Path:
    """Resolve and validate a local media path."""
    path = pathlib.Path(file_path).expanduser().resolve()
    if not path.exists():
        print(f"ERROR: media path does not exist: {file_path}", file=sys.stderr)
        sys.exit(1)
    if not path.is_file():
        print(f"ERROR: media path is not a file: {file_path}", file=sys.stderr)
        sys.exit(1)
    return path


def is_supported_mime(mime: str) -> bool:
    """Return True if the CLI accepts this MIME type."""
    return (
        mime.startswith(_SUPPORTED_MEDIA_PREFIXES)
        or mime in _SUPPORTED_EXTRA_MIME_TYPES
    )


def normalize_state(state: Any) -> str:
    """Normalize SDK file state strings/enums."""
    if hasattr(state, "name"):
        return str(state.name).upper()
    return str(state).upper()


def upload_media(client, path: pathlib.Path, mime: str, timeout_seconds: int):
    """Upload media to Gemini Files API and wait until ACTIVE."""
    try:
        file_obj = client.files.upload(file=str(path), config={"mime_type": mime})
        waited = 0
        while True:
            latest = client.files.get(name=file_obj.name)
            state = normalize_state(getattr(latest, "state", "UNKNOWN"))
            if state == "ACTIVE":
                return latest
            if state in {"FAILED", "ERROR"}:
                print(
                    f"ERROR: media upload processing failed: state={state}",
                    file=sys.stderr,
                )
                sys.exit(1)
            if waited >= timeout_seconds:
                print(
                    "ERROR: media upload did not become ACTIVE within "
                    f"{timeout_seconds}s; file={file_obj.name}",
                    file=sys.stderr,
                )
                sys.exit(1)
            time.sleep(_POLL_INTERVAL_SECONDS)
            waited += _POLL_INTERVAL_SECONDS
    except SystemExit:
        raise
    except Exception as exc:
        print(f"ERROR: failed to upload media: {exc}", file=sys.stderr)
        sys.exit(1)


def build_prompt(path: pathlib.Path, mime: str, user_prompt: str | None) -> str:
    """Build objective media-analysis prompt."""
    schema = json.dumps(_BASE_SCHEMA, indent=2)
    extra = f"\nAdditional user request: {user_prompt}\n" if user_prompt else ""
    return f"""Analyze this media file objectively.

File name: {path.name}
MIME type: {mime}

Required behavior:
- Transcribe spoken audio with timestamps when possible.
- Translate speech to English when possible.
- Extract visible on-screen text with timestamps when possible.
- Describe visual scenes/events objectively when video or image content is present.
- Do not fact-check, judge, classify, or interpret claims.
- Do not use external sources.
- Use null/empty arrays when a field is not applicable or cannot be determined.
{extra}
Return JSON matching this shape:
{schema}
"""


def extract_response_text(response) -> str:
    """Extract text from a Gemini response."""
    text = getattr(response, "text", None)
    if text:
        return text
    candidates = getattr(response, "candidates", None) or []
    chunks: list[str] = []
    for candidate in candidates:
        content = getattr(candidate, "content", None)
        parts = getattr(content, "parts", None) if content else None
        for part in parts or []:
            part_text = getattr(part, "text", None)
            if part_text:
                chunks.append(part_text)
    return "\n".join(chunks)


def parse_json_response(text: str) -> dict[str, Any] | None:
    """Best-effort parse of model JSON response."""
    stripped = text.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        stripped = "\n".join(lines).strip()
    try:
        parsed = json.loads(stripped)
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


def analyze_media(
    media_path: str,
    *,
    model: str = _DEFAULT_MODEL,
    prompt: str | None = None,
    timeout_seconds: int = _DEFAULT_TIMEOUT_SECONDS,
    as_json: bool = False,
) -> dict[str, Any]:
    """Upload and objectively analyze audio/video/image media."""
    path = validate_media_path(media_path)
    mime = detect_mime(path)
    if not is_supported_mime(mime):
        print(f"ERROR: unsupported media MIME type: {mime}", file=sys.stderr)
        sys.exit(1)

    key = get_api_key()
    client = make_client(key)
    uploaded = upload_media(client, path, mime, timeout_seconds)

    from google.genai import types

    contents = [
        types.Part.from_uri(file_uri=uploaded.uri, mime_type=mime),
        build_prompt(path, mime, prompt),
    ]
    config = types.GenerateContentConfig(
        system_instruction=_OBJECTIVE_SYSTEM_INSTRUCTION,
        response_mime_type="application/json",
    )

    try:
        response = client.models.generate_content(
            model=model,
            contents=contents,
            config=config,
        )
    except Exception as exc:
        print(f"ERROR: Gemini media analysis failed: {exc}", file=sys.stderr)
        sys.exit(1)

    raw_text = extract_response_text(response)
    parsed = parse_json_response(raw_text)
    if parsed is None:
        result: dict[str, Any] = {
            "media": {"path": str(path), "mime_type": mime, "duration_seconds": None},
            "model": model,
            "raw_text": raw_text,
            "parse_error": "model response was not valid JSON",
        }
    else:
        result = parsed
        result.setdefault("media", {})
        result["media"].setdefault("path", str(path))
        result["media"].setdefault("mime_type", mime)
        result.setdefault("model", model)

    if as_json:
        print(json.dumps(result, indent=2, ensure_ascii=False))
    else:
        print_human(result)
    return result


def print_human(result: dict[str, Any]) -> None:
    """Print a compact human-readable summary."""
    media = result.get("media", {})
    print(f"Media: {media.get('path', 'unknown')}")
    print(f"MIME: {media.get('mime_type', 'unknown')}")
    print(f"Model: {result.get('model', 'unknown')}")
    print()

    audio = result.get("audio", {}) or {}
    transcript = audio.get("transcript", []) or []
    print("=== TRANSCRIPT ===")
    if transcript:
        for item in transcript:
            start = item.get("start") or "?"
            end = item.get("end") or "?"
            text = item.get("text") or ""
            translation = item.get("translation_en")
            print(f"[{start} - {end}] {text}")
            if translation:
                print(f"  EN: {translation}")
    else:
        print("No transcript returned.")

    visual = result.get("visual", {}) or {}
    onscreen_text = visual.get("onscreen_text", []) or []
    scenes = visual.get("scene_observations", []) or []
    print()
    print("=== ONSCREEN TEXT ===")
    if onscreen_text:
        for item in onscreen_text:
            ts = item.get("timestamp") or "?"
            print(f"[{ts}] {item.get('text', '')}")
    else:
        print("No on-screen text returned.")

    print()
    print("=== SCENE OBSERVATIONS ===")
    if scenes:
        for item in scenes:
            ts = item.get("timestamp") or "?"
            print(f"[{ts}] {item.get('description', '')}")
    else:
        print("No scene observations returned.")

    if "raw_text" in result:
        print()
        print("=== RAW MODEL TEXT ===")
        print(result["raw_text"])


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Objective Gemini media transcription and observation CLI"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    analyze = subparsers.add_parser(
        "analyze", help="Transcribe/observe audio, video, image, or PDF media"
    )
    analyze.add_argument("media_path", help="Path to local media file")
    analyze.add_argument("--model", default=_DEFAULT_MODEL, help="Gemini model")
    analyze.add_argument(
        "--prompt",
        help="Additional objective transcription/observation request; no fact-checking",
    )
    analyze.add_argument("--json", action="store_true", help="Print JSON output")
    analyze.add_argument(
        "--timeout-seconds",
        type=int,
        default=_DEFAULT_TIMEOUT_SECONDS,
        help="Seconds to wait for Files API processing",
    )

    args = parser.parse_args()
    if args.command == "analyze":
        analyze_media(
            args.media_path,
            model=args.model,
            prompt=args.prompt,
            timeout_seconds=args.timeout_seconds,
            as_json=args.json,
        )
    else:  # pragma: no cover - argparse prevents this
        parser.error(f"unknown command: {args.command}")


if __name__ == "__main__":
    main()
