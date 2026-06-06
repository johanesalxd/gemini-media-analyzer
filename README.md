# Gemini Media Analyzer

Objective Gemini audio/video/image transcription and observation CLI.

This repo is intentionally boring. It uploads media to Gemini, asks for direct transcription and observable scene/text notes, then returns structured JSON. It does **not** fact-check, classify propaganda, infer intent, judge truth, or perform web research.

## What it does

- Transcribes spoken audio with timestamps when Gemini can provide them.
- Translates speech to English when possible.
- Extracts visible on-screen text.
- Describes visible scenes/events objectively.
- Marks uncertainty instead of filling gaps with vibes.

## What it does not do

- No fact-checking.
- No misinformation verdicts.
- No propaganda/bias classification.
- No political interpretation.
- No external-source research.
- No hidden-intent analysis.

Use another tool/operator to decide whether claims are true.

## Setup

Requires Python 3.13+ and a Gemini API key.

```bash
uv sync
export GOOGLE_API_KEY="..."
```

`GEMINI_API_KEY` is accepted as a fallback, but `GOOGLE_API_KEY` is preferred.

## Usage

```bash
uv run python gemini_media_analyzer.py analyze ./clip.mp4 --json
uv run python gemini_media_analyzer.py analyze ./audio.wav --json
uv run python gemini_media_analyzer.py analyze ./frame.jpg --json
```

Optional model:

```bash
uv run python gemini_media_analyzer.py analyze ./clip.mp4 \
  --model gemini-3.5-flash \
  --json
```

Optional objective prompt:

```bash
uv run python gemini_media_analyzer.py analyze ./clip.mp4 \
  --prompt "Focus on timestamped on-screen text and speech transcription." \
  --json
```

Do not put fact-checking or judgment requests in `--prompt`; the system instruction tells the model to refuse that role and stay observational.

## Output shape

```json
{
  "media": {
    "path": "...",
    "mime_type": "video/mp4",
    "duration_seconds": null
  },
  "audio": {
    "detected_languages": ["id"],
    "transcript": [
      {
        "start": "00:00:00",
        "end": "00:00:08",
        "text": "...",
        "translation_en": "...",
        "confidence": "medium"
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
        "description": "A digitally rendered boardroom scene with people seated around a table."
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

## Public-safety notes

- Never commit `.env`, API keys, media cache files, transcripts containing private content, or generated analysis outputs unless intentionally public.
- This repo has no bundled credentials and no user-specific paths.
- Media is uploaded to Gemini's Files API for processing; do not use it for private media unless that upload is acceptable.

## Development

```bash
uv run pytest
uv run ruff check .
uv run python gemini_media_analyzer.py --help
```
