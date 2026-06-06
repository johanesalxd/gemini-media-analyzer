---
name: gemini-media-analyzer
description: Use when you need objective Gemini-based audio, video, image, or PDF transcription, translation, OCR, and scene observation without fact-checking or judgment.
version: 1.0.0
author: Hermes Agent
license: MIT
metadata:
  hermes:
    tags: [gemini, media, video, audio, transcription, ocr, multimodal]
    related_skills: []
---

# Gemini Media Analyzer

## Overview

Use this skill to run `gemini-media-analyzer`, a deliberately narrow CLI for Gemini-native media perception. It uploads a local audio, video, image, or PDF file to Gemini, requests objective transcription/translation/OCR/scene notes, validates the result, and returns structured JSON.

This is an eyes-and-ears tool. It does not fact-check claims, classify propaganda or misinformation, infer motives, perform web research, or decide whether media is true. Use a separate research or fact-checking workflow for those judgments.

## When to Use

Use when:

- You need a timestamped transcript of audio or video.
- You need English translation alongside non-English speech.
- You need on-screen text/OCR from video or image content.
- You need objective scene observations from visual media.
- You need structured JSON for a downstream workflow.
- Uploading the media to Gemini is acceptable.

Do not use when:

- The task requires truth verification, source research, or a final verdict.
- The media is private/sensitive and Gemini upload is not acceptable.
- You need forensic-grade repeatability from local-only tools.
- A simple local transcript is enough and cloud processing is unnecessary.

## Setup

From the repository root:

```bash
uv sync
cp .env.example .env
# edit .env and set GOOGLE_API_KEY
```

`GOOGLE_API_KEY` is preferred. `GEMINI_API_KEY` is accepted as a fallback. The CLI automatically loads `.env` from the current working directory and does not override variables already present in the environment.

## Core Commands

Analyze a video and print JSON:

```bash
uv run python gemini_media_analyzer.py analyze ./clip.mp4 --json
```

Analyze audio:

```bash
uv run python gemini_media_analyzer.py analyze ./audio.wav --json
```

Analyze an image:

```bash
uv run python gemini_media_analyzer.py analyze ./frame.jpg --json
```

Write JSON to a file:

```bash
uv run python gemini_media_analyzer.py analyze ./clip.mp4 \
  --output ./analysis.json
```

Use an explicit model:

```bash
uv run python gemini_media_analyzer.py analyze ./clip.mp4 \
  --model gemini-3.5-flash \
  --json
```

Add an objective prompt:

```bash
uv run python gemini_media_analyzer.py analyze ./clip.mp4 \
  --prompt "Focus on timestamped on-screen text and speech transcription." \
  --json
```

Keep the Gemini Files API upload instead of deleting it after analysis:

```bash
uv run python gemini_media_analyzer.py analyze ./clip.mp4 \
  --keep-uploaded-file \
  --json
```

Use `--keep-uploaded-file` only when you explicitly need to inspect or reuse the uploaded file. The default cleanup behavior is safer.

## Output Contract

Normal successful output is JSON with this shape:

```json
{
  "media": {
    "path": "clip.mp4",
    "mime_type": "video/mp4",
    "duration_seconds": 123.4
  },
  "audio": {
    "detected_languages": ["id"],
    "transcript": [
      {
        "start": "00:00:00",
        "end": "00:00:08",
        "text": "...",
        "translation_en": "...",
        "confidence": "high"
      }
    ],
    "unclear_segments": []
  },
  "visual": {
    "onscreen_text": [
      {
        "timestamp": "00:00:30",
        "text": "...",
        "confidence": "high"
      }
    ],
    "scene_observations": [
      {
        "timestamp": "00:00:30",
        "description": "Objective description of visible scene or event."
      }
    ]
  },
  "model_notes": {
    "limitations": [],
    "safety_blocks": []
  },
  "model": "gemini-3.5-flash"
}
```

If Gemini returns malformed JSON or a response that fails schema validation, the CLI returns a structured fallback containing `raw_text`, `parse_error`, and/or `validation_error` instead of silently pretending the result is clean.

## Perception-Only Boundary

The system instruction and prompt are designed to keep the model observational:

- transcribe speech
- translate speech
- extract visible text
- describe visible scenes/events
- mark uncertainty

Do not ask this tool to:

- fact-check claims
- decide whether claims are true or false
- classify misinformation, propaganda, bias, or ideology
- infer intent or hidden meaning
- perform external-source research
- produce political analysis

If you need those outputs, use this tool first for perception, then pass the resulting transcript/OCR/scene notes to a separate research or fact-checking process.

## Privacy and Upload Warning

This CLI uploads media to Gemini's Files API. Do not use it for private, confidential, regulated, or user-sensitive media unless that upload is acceptable.

Do not commit:

- `.env`
- API keys
- uploaded media
- transcripts containing private content
- generated JSON outputs

`.env.example` is safe to commit; `.env` is not.

## Development Checks

Run these before publishing changes:

```bash
uv run pytest
uv run ruff check .
uv run python gemini_media_analyzer.py --help
uv run python gemini_media_analyzer.py analyze --help
```

For a live smoke test, use a small non-sensitive media file:

```bash
uv run python gemini_media_analyzer.py analyze ./sample.mp4 \
  --output /tmp/gemini-media-smoke.json
```

Then inspect the output for:

- valid JSON
- correct `media.mime_type`
- expected `model`
- transcript/OCR/scene fields present where applicable
- no unwanted fact-check or judgment language

## Common Pitfalls

1. **Treating output as truth.** The tool reports what Gemini perceived, not whether claims are correct.
2. **Using it on sensitive media.** Media is uploaded to Gemini; choose local-only tools when upload is not acceptable.
3. **Putting judgment into `--prompt`.** Keep prompts observational. Do not ask for misinformation or propaganda classification.
4. **Ignoring fallback fields.** If `parse_error` or `validation_error` exists, treat the result as degraded and inspect `raw_text`.
5. **Forgetting cleanup semantics.** Uploaded files are deleted by default after analysis. Use `--keep-uploaded-file` only when necessary.
6. **Committing generated artifacts.** Keep `.env`, media, transcripts, and analysis outputs out of public repositories unless intentionally published.

## Verification Checklist

- [ ] `.env` exists locally or `GOOGLE_API_KEY`/`GEMINI_API_KEY` is exported.
- [ ] `uv sync` completed.
- [ ] `uv run python gemini_media_analyzer.py analyze --help` works.
- [ ] Live media command returns valid JSON or an explicit fallback object.
- [ ] Output stays observational: transcript, translation, OCR, scene notes, uncertainty.
- [ ] No fact-checking, propaganda classification, or truth verdict appears in the tool output.
- [ ] Sensitive media and generated outputs are not committed.
