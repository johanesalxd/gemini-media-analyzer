"""Unit tests for gemini_media_analyzer."""

import json
from types import SimpleNamespace

import pytest

import gemini_media_analyzer as gma


def test_get_api_key_prefers_google(monkeypatch):
    monkeypatch.setenv("GOOGLE_API_KEY", "google-key")
    monkeypatch.setenv("GEMINI_API_KEY", "gemini-key")

    assert gma.get_api_key() == "google-key"


def test_get_api_key_accepts_gemini_fallback(monkeypatch):
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.setenv("GEMINI_API_KEY", "gemini-key")

    assert gma.get_api_key() == "gemini-key"


def test_get_api_key_exits_when_missing(monkeypatch):
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)

    with pytest.raises(SystemExit) as exc:
        gma.get_api_key()

    assert exc.value.code == 1


@pytest.mark.parametrize(
    ("mime", "expected"),
    [
        ("video/mp4", True),
        ("audio/wav", True),
        ("image/jpeg", True),
        ("application/pdf", True),
        ("text/plain", False),
        ("application/octet-stream", False),
    ],
)
def test_is_supported_mime(mime, expected):
    assert gma.is_supported_mime(mime) is expected


def test_build_prompt_forbids_judgment(tmp_path):
    media = tmp_path / "clip.mp4"
    media.write_bytes(b"fake")

    prompt = gma.build_prompt(media, "video/mp4", "Focus on OCR")

    assert "Do not fact-check" in prompt
    assert "Do not use external sources" in prompt
    assert "Focus on OCR" in prompt
    assert "Return JSON" in prompt


def test_parse_json_response_handles_plain_json():
    assert gma.parse_json_response('{"ok": true}') == {"ok": True}


def test_parse_json_response_handles_markdown_fence():
    assert gma.parse_json_response('```json\n{"ok": true}\n```') == {"ok": True}


def test_parse_json_response_returns_none_for_invalid():
    assert gma.parse_json_response("not json") is None


def test_upload_media_polls_until_active(mocker, tmp_path):
    media = tmp_path / "clip.mp4"
    media.write_bytes(b"fake")
    uploaded = SimpleNamespace(name="files/abc", uri="gemini://abc")
    processing = SimpleNamespace(
        name="files/abc", uri="gemini://abc", state="PROCESSING"
    )
    active = SimpleNamespace(name="files/abc", uri="gemini://abc", state="ACTIVE")
    client = SimpleNamespace(
        files=SimpleNamespace(
            upload=mocker.Mock(return_value=uploaded),
            get=mocker.Mock(side_effect=[processing, active]),
        )
    )
    mocker.patch("time.sleep")

    result = gma.upload_media(client, media, "video/mp4", timeout_seconds=10)

    assert result is active
    client.files.upload.assert_called_once_with(
        file=str(media), config={"mime_type": "video/mp4"}
    )


def test_main_dispatch_analyze(mocker):
    mock_analyze = mocker.patch("gemini_media_analyzer.analyze_media")
    mocker.patch(
        "sys.argv",
        [
            "gemini_media_analyzer.py",
            "analyze",
            "clip.mp4",
            "--model",
            "gemini-3.5-flash",
            "--prompt",
            "OCR only",
            "--json",
            "--timeout-seconds",
            "42",
        ],
    )

    gma.main()

    mock_analyze.assert_called_once_with(
        "clip.mp4",
        model="gemini-3.5-flash",
        prompt="OCR only",
        timeout_seconds=42,
        as_json=True,
    )


def test_print_json_when_model_response_valid(mocker, tmp_path, monkeypatch, capsys):
    media = tmp_path / "clip.mp4"
    media.write_bytes(b"fake")
    monkeypatch.setenv("GOOGLE_API_KEY", "key")
    active = SimpleNamespace(uri="gemini://abc")
    client = SimpleNamespace(
        models=SimpleNamespace(
            generate_content=mocker.Mock(
                return_value=SimpleNamespace(
                    text=json.dumps({"audio": {"transcript": []}, "visual": {}})
                )
            )
        )
    )
    mocker.patch("gemini_media_analyzer.make_client", return_value=client)
    mocker.patch("gemini_media_analyzer.upload_media", return_value=active)

    result = gma.analyze_media(str(media), as_json=True)

    captured = capsys.readouterr()
    assert '"audio"' in captured.out
    assert result["media"]["mime_type"] == "video/mp4"
    assert result["model"] == gma._DEFAULT_MODEL
