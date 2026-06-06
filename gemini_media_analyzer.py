#!/usr/bin/env python3
"""Objective Gemini media transcription/observation CLI.

This tool is deliberately dumb: it transcribes and describes media, but does not
fact-check, classify propaganda, infer intent, or decide truth.

Env: GOOGLE_API_KEY (required); GEMINI_API_KEY is accepted as a fallback.
The CLI loads a local .env file automatically when present.
"""

from __future__ import annotations

import argparse
import json
import mimetypes
import os
import pathlib
import sys
import tempfile
import time
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

_DEFAULT_MODEL = "gemini-3.5-flash"
_DEFAULT_TIMEOUT_SECONDS = 300
_POLL_INTERVAL_SECONDS = 2
_SUPPORTED_MEDIA_PREFIXES = ("audio/", "video/", "image/")
_SUPPORTED_EXTRA_MIME_TYPES = {"application/pdf"}
_ENV_FILE_NAME = ".env"

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

Confidence = Literal["high", "medium", "low", "unknown"]
UnclearReason = Literal["inaudible", "overlap", "music", "other"]


class GeminiMediaAnalyzerError(Exception):
    """Base class for user-facing analyzer errors."""


class ConfigError(GeminiMediaAnalyzerError):
    """Configuration or environment error."""


class MediaValidationError(GeminiMediaAnalyzerError):
    """Invalid local media input."""


class UploadError(GeminiMediaAnalyzerError):
    """Gemini Files API upload or cleanup error."""


class AnalysisError(GeminiMediaAnalyzerError):
    """Gemini model analysis error."""


class OutputError(GeminiMediaAnalyzerError):
    """Output writing error."""


class AnalyzerModel(BaseModel):
    """Shared Pydantic settings for model output validation."""

    model_config = ConfigDict(extra="ignore")


class MediaInfo(AnalyzerModel):
    path: str
    mime_type: str
    duration_seconds: float | None = None


class TranscriptSegment(AnalyzerModel):
    start: str | None = None
    end: str | None = None
    text: str
    translation_en: str | None = None
    confidence: Confidence = "unknown"


class UnclearSegment(AnalyzerModel):
    timestamp: str | None = None
    reason: UnclearReason = "other"
    best_effort_text: str | None = None


class AudioInfo(AnalyzerModel):
    detected_languages: list[str] = Field(default_factory=list)
    transcript: list[TranscriptSegment] = Field(default_factory=list)
    unclear_segments: list[UnclearSegment] = Field(default_factory=list)


class OnscreenText(AnalyzerModel):
    timestamp: str | None = None
    text: str
    confidence: Confidence = "unknown"


class SceneObservation(AnalyzerModel):
    timestamp: str | None = None
    description: str


class VisualInfo(AnalyzerModel):
    onscreen_text: list[OnscreenText] = Field(default_factory=list)
    scene_observations: list[SceneObservation] = Field(default_factory=list)


class ModelNotes(AnalyzerModel):
    limitations: list[str] = Field(default_factory=list)
    safety_blocks: list[str] = Field(default_factory=list)


class AnalysisResult(AnalyzerModel):
    media: MediaInfo
    audio: AudioInfo = Field(default_factory=AudioInfo)
    visual: VisualInfo = Field(default_factory=VisualInfo)
    model_notes: ModelNotes = Field(default_factory=ModelNotes)
    model: str


def load_dotenv(path: pathlib.Path | None = None) -> None:
    """Load simple KEY=VALUE lines from .env without overriding the environment."""
    env_path = path or pathlib.Path.cwd() / _ENV_FILE_NAME
    if not env_path.exists():
        return
    try:
        lines = env_path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise ConfigError(f"failed to read {env_path}: {exc}") from exc

    for raw_line in lines:
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def get_api_key() -> str:
    """Read Gemini API key from environment."""
    load_dotenv()
    key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
    if not key:
        raise ConfigError(
            "GOOGLE_API_KEY environment variable not set "
            "(GEMINI_API_KEY fallback also empty)"
        )
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
        raise MediaValidationError(f"media path does not exist: {file_path}")
    if not path.is_file():
        raise MediaValidationError(f"media path is not a file: {file_path}")
    return path


def validate_timeout_seconds(timeout_seconds: int) -> int:
    """Return timeout when positive, otherwise raise."""
    if timeout_seconds <= 0:
        raise MediaValidationError("--timeout-seconds must be greater than 0")
    return timeout_seconds


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
    timeout_seconds = validate_timeout_seconds(timeout_seconds)
    try:
        file_obj = client.files.upload(file=str(path), config={"mime_type": mime})
        waited = 0
        while True:
            latest = client.files.get(name=file_obj.name)
            state = normalize_state(getattr(latest, "state", "UNKNOWN"))
            if state == "ACTIVE":
                return latest
            if state in {"FAILED", "ERROR"}:
                raise UploadError(f"media upload processing failed: state={state}")
            if waited >= timeout_seconds:
                raise UploadError(
                    "media upload did not become ACTIVE within "
                    f"{timeout_seconds}s; file={file_obj.name}"
                )
            time.sleep(_POLL_INTERVAL_SECONDS)
            waited += _POLL_INTERVAL_SECONDS
    except GeminiMediaAnalyzerError:
        raise
    except Exception as exc:
        raise UploadError(f"failed to upload media: {exc}") from exc


def delete_uploaded_file(client, uploaded_file) -> None:
    """Delete a Gemini Files API upload."""
    name = getattr(uploaded_file, "name", None)
    if not name:
        raise UploadError("uploaded file has no name; cannot delete it")
    try:
        client.files.delete(name=name)
    except Exception as exc:
        raise UploadError(f"failed to delete uploaded file {name}: {exc}") from exc


def build_prompt(path: pathlib.Path, mime: str, user_prompt: str | None) -> str:
    """Build objective media-analysis prompt."""
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
Return JSON matching the configured response schema.
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


def normalize_analysis_result(
    parsed: dict[str, Any],
    *,
    path: pathlib.Path,
    mime: str,
    model: str,
) -> dict[str, Any]:
    """Validate and normalize model output."""
    candidate = dict(parsed)
    media = candidate.get("media")
    if not isinstance(media, dict):
        media = {}
    candidate["media"] = {
        "path": str(path),
        "mime_type": mime,
        **media,
    }
    candidate["media"]["path"] = candidate["media"].get("path") or str(path)
    candidate["media"]["mime_type"] = candidate["media"].get("mime_type") or mime
    candidate["model"] = candidate.get("model") or model
    result = AnalysisResult.model_validate(candidate)
    return result.model_dump(mode="json")


def fallback_result(
    *,
    path: pathlib.Path,
    mime: str,
    model: str,
    raw_text: str,
    parse_error: str | None = None,
    validation_error: str | None = None,
) -> dict[str, Any]:
    """Build a structured fallback for malformed model responses."""
    return {
        "media": {"path": str(path), "mime_type": mime, "duration_seconds": None},
        "audio": {"detected_languages": [], "transcript": [], "unclear_segments": []},
        "visual": {"onscreen_text": [], "scene_observations": []},
        "model_notes": {"limitations": [], "safety_blocks": []},
        "model": model,
        "raw_text": raw_text,
        "parse_error": parse_error,
        "validation_error": validation_error,
    }


def write_json_output(result: dict[str, Any], output_path: str) -> None:
    """Atomically write JSON output to a local path."""
    path = pathlib.Path(output_path).expanduser()
    directory = path.parent if path.parent != pathlib.Path("") else pathlib.Path.cwd()
    if not directory.exists():
        raise OutputError(f"output directory does not exist: {directory}")
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            dir=directory,
            delete=False,
            prefix=f".{path.name}.",
            suffix=".tmp",
        ) as temp_file:
            temp_file.write(json.dumps(result, indent=2, ensure_ascii=False))
            temp_file.write("\n")
            temp_name = temp_file.name
        pathlib.Path(temp_name).replace(path)
    except OSError as exc:
        raise OutputError(f"failed to write output JSON to {path}: {exc}") from exc


def analyze_media(
    media_path: str,
    *,
    model: str = _DEFAULT_MODEL,
    prompt: str | None = None,
    timeout_seconds: int = _DEFAULT_TIMEOUT_SECONDS,
    as_json: bool = False,
    output_path: str | None = None,
    keep_uploaded_file: bool = False,
) -> dict[str, Any]:
    """Upload and objectively analyze audio/video/image media."""
    path = validate_media_path(media_path)
    validate_timeout_seconds(timeout_seconds)
    mime = detect_mime(path)
    if not is_supported_mime(mime):
        raise MediaValidationError(f"unsupported media MIME type: {mime}")

    key = get_api_key()
    client = make_client(key)
    uploaded = None
    operation_error: GeminiMediaAnalyzerError | None = None
    result: dict[str, Any] | None = None

    from google.genai import types

    try:
        uploaded = upload_media(client, path, mime, timeout_seconds)
        contents = [
            types.Part.from_uri(file_uri=uploaded.uri, mime_type=mime),
            build_prompt(path, mime, prompt),
        ]
        config = types.GenerateContentConfig(
            system_instruction=_OBJECTIVE_SYSTEM_INSTRUCTION,
            response_mime_type="application/json",
            response_json_schema=AnalysisResult.model_json_schema(),
        )

        response = client.models.generate_content(
            model=model,
            contents=contents,
            config=config,
        )

        raw_text = extract_response_text(response)
        parsed = parse_json_response(raw_text)
        if parsed is None:
            result = fallback_result(
                path=path,
                mime=mime,
                model=model,
                raw_text=raw_text,
                parse_error="model response was not valid JSON object",
            )
        else:
            try:
                result = normalize_analysis_result(
                    parsed,
                    path=path,
                    mime=mime,
                    model=model,
                )
            except ValidationError as exc:
                result = fallback_result(
                    path=path,
                    mime=mime,
                    model=model,
                    raw_text=raw_text,
                    validation_error=str(exc),
                )
    except GeminiMediaAnalyzerError as exc:
        operation_error = exc
    except Exception as exc:
        operation_error = AnalysisError(f"Gemini media analysis failed: {exc}")
    finally:
        if uploaded is not None and not keep_uploaded_file:
            try:
                delete_uploaded_file(client, uploaded)
            except UploadError as exc:
                if operation_error is None:
                    operation_error = exc

    if operation_error is not None:
        raise operation_error
    if result is None:
        raise AnalysisError("Gemini media analysis did not produce a result")
    if output_path:
        write_json_output(result, output_path)
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
    analyze.add_argument("--output", help="Write JSON output to this path")
    analyze.add_argument(
        "--keep-uploaded-file",
        action="store_true",
        help="Do not delete the Gemini Files API upload after analysis",
    )
    analyze.add_argument(
        "--timeout-seconds",
        type=int,
        default=_DEFAULT_TIMEOUT_SECONDS,
        help="Seconds to wait for Files API processing",
    )

    args = parser.parse_args()
    try:
        if args.command == "analyze":
            analyze_media(
                args.media_path,
                model=args.model,
                prompt=args.prompt,
                timeout_seconds=args.timeout_seconds,
                as_json=args.json,
                output_path=args.output,
                keep_uploaded_file=args.keep_uploaded_file,
            )
        else:  # pragma: no cover - argparse prevents this
            parser.error(f"unknown command: {args.command}")
    except GeminiMediaAnalyzerError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
