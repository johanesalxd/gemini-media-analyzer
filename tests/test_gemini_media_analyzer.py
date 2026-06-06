"""Unit tests for gemini_media_analyzer."""

import json
from types import SimpleNamespace

import pytest

import gemini_media_analyzer as gma


def make_valid_response_text() -> str:
    return json.dumps(
        {
            "model": "hallucinated-model-name",
            "audio": {
                "detected_languages": ["en"],
                "transcript": [
                    {
                        "start": "00:00:00",
                        "end": "00:00:02",
                        "text": "Hello",
                        "translation_en": None,
                        "confidence": "high",
                    }
                ],
                "unclear_segments": [],
            },
            "visual": {
                "onscreen_text": [],
                "scene_observations": [
                    {
                        "timestamp": None,
                        "description": "A title card is visible.",
                    }
                ],
            },
            "model_notes": {"limitations": [], "safety_blocks": []},
        }
    )


def make_client(mocker, response_text: str | None = None):
    generate_content = mocker.Mock(return_value=SimpleNamespace(text=response_text))
    return SimpleNamespace(
        files=SimpleNamespace(delete=mocker.Mock()),
        models=SimpleNamespace(generate_content=generate_content),
    )


def test_get_api_key_prefers_google(monkeypatch):
    monkeypatch.setattr(gma, "load_dotenv", lambda: None)
    monkeypatch.setenv("GOOGLE_API_KEY", "google-key")
    monkeypatch.setenv("GEMINI_API_KEY", "gemini-key")

    assert gma.get_api_key() == "google-key"


def test_get_api_key_accepts_gemini_fallback(monkeypatch):
    monkeypatch.setattr(gma, "load_dotenv", lambda: None)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.setenv("GEMINI_API_KEY", "gemini-key")

    assert gma.get_api_key() == "gemini-key"


def test_get_api_key_raises_when_missing(monkeypatch):
    monkeypatch.setattr(gma, "load_dotenv", lambda: None)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)

    with pytest.raises(gma.ConfigError):
        gma.get_api_key()


def test_load_dotenv_loads_values_without_overriding(tmp_path, monkeypatch):
    env_file = tmp_path / ".env"
    env_file.write_text(
        "# comment\nGOOGLE_API_KEY=from-dotenv\nGEMINI_API_KEY='fallback'\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("GOOGLE_API_KEY", "from-real-env")
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)

    gma.load_dotenv(env_file)

    assert gma.os.environ["GOOGLE_API_KEY"] == "from-real-env"
    assert gma.os.environ["GEMINI_API_KEY"] == "fallback"


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
    assert "configured response schema" in prompt


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


def test_upload_media_raises_for_failed_state(mocker, tmp_path):
    media = tmp_path / "clip.mp4"
    media.write_bytes(b"fake")
    uploaded = SimpleNamespace(name="files/abc", uri="gemini://abc")
    failed = SimpleNamespace(name="files/abc", uri="gemini://abc", state="FAILED")
    client = SimpleNamespace(
        files=SimpleNamespace(
            upload=mocker.Mock(return_value=uploaded),
            get=mocker.Mock(return_value=failed),
        )
    )

    with pytest.raises(gma.UploadError, match="state=FAILED"):
        gma.upload_media(client, media, "video/mp4", timeout_seconds=10)


def test_upload_media_raises_on_timeout(mocker, tmp_path):
    media = tmp_path / "clip.mp4"
    media.write_bytes(b"fake")
    uploaded = SimpleNamespace(name="files/abc", uri="gemini://abc")
    processing = SimpleNamespace(
        name="files/abc", uri="gemini://abc", state="PROCESSING"
    )
    client = SimpleNamespace(
        files=SimpleNamespace(
            upload=mocker.Mock(return_value=uploaded),
            get=mocker.Mock(return_value=processing),
        )
    )
    mocker.patch("time.sleep")

    with pytest.raises(gma.UploadError, match="did not become ACTIVE"):
        gma.upload_media(client, media, "video/mp4", timeout_seconds=1)


def test_validate_timeout_seconds_requires_positive_value():
    with pytest.raises(gma.MediaValidationError):
        gma.validate_timeout_seconds(0)


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
            "--output",
            "out.json",
            "--keep-uploaded-file",
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
        output_path="out.json",
        keep_uploaded_file=True,
    )


def test_analyze_media_validates_and_deletes_upload(
    mocker, tmp_path, monkeypatch, capsys
):
    media = tmp_path / "clip.mp4"
    media.write_bytes(b"fake")
    monkeypatch.setenv("GOOGLE_API_KEY", "key")
    active = SimpleNamespace(name="files/abc", uri="gemini://abc")
    client = make_client(mocker, make_valid_response_text())
    mocker.patch("gemini_media_analyzer.make_client", return_value=client)
    mocker.patch("gemini_media_analyzer.upload_media", return_value=active)

    result = gma.analyze_media(str(media), as_json=True)

    captured = capsys.readouterr()
    assert '"audio"' in captured.out
    assert result["media"]["mime_type"] == "video/mp4"
    assert result["model"] == gma._DEFAULT_MODEL
    assert result["audio"]["transcript"][0]["text"] == "Hello"
    client.files.delete.assert_called_once_with(name="files/abc")


def test_analyze_media_deletes_upload_when_generation_fails(
    mocker, tmp_path, monkeypatch
):
    media = tmp_path / "clip.mp4"
    media.write_bytes(b"fake")
    monkeypatch.setenv("GOOGLE_API_KEY", "key")
    active = SimpleNamespace(name="files/abc", uri="gemini://abc")
    client = make_client(mocker)
    client.models.generate_content.side_effect = RuntimeError("boom")
    mocker.patch("gemini_media_analyzer.make_client", return_value=client)
    mocker.patch("gemini_media_analyzer.upload_media", return_value=active)

    with pytest.raises(gma.AnalysisError, match="boom"):
        gma.analyze_media(str(media), as_json=True)

    client.files.delete.assert_called_once_with(name="files/abc")


def test_analyze_media_can_keep_uploaded_file(mocker, tmp_path, monkeypatch):
    media = tmp_path / "clip.mp4"
    media.write_bytes(b"fake")
    monkeypatch.setenv("GOOGLE_API_KEY", "key")
    active = SimpleNamespace(name="files/abc", uri="gemini://abc")
    client = make_client(mocker, make_valid_response_text())
    mocker.patch("gemini_media_analyzer.make_client", return_value=client)
    mocker.patch("gemini_media_analyzer.upload_media", return_value=active)

    gma.analyze_media(str(media), as_json=True, keep_uploaded_file=True)

    client.files.delete.assert_not_called()


def test_analyze_media_returns_validation_fallback(mocker, tmp_path, monkeypatch):
    media = tmp_path / "clip.mp4"
    media.write_bytes(b"fake")
    monkeypatch.setenv("GOOGLE_API_KEY", "key")
    active = SimpleNamespace(name="files/abc", uri="gemini://abc")
    invalid_response = json.dumps({"audio": {"transcript": [{"confidence": "high"}]}})
    client = make_client(mocker, invalid_response)
    mocker.patch("gemini_media_analyzer.make_client", return_value=client)
    mocker.patch("gemini_media_analyzer.upload_media", return_value=active)

    result = gma.analyze_media(str(media), as_json=True)

    assert result["parse_error"] is None
    assert "validation_error" in result
    assert "text" in result["validation_error"]
    assert result["raw_text"] == invalid_response
    client.files.delete.assert_called_once_with(name="files/abc")


def test_analyze_media_returns_parse_fallback(mocker, tmp_path, monkeypatch):
    media = tmp_path / "clip.mp4"
    media.write_bytes(b"fake")
    monkeypatch.setenv("GOOGLE_API_KEY", "key")
    active = SimpleNamespace(name="files/abc", uri="gemini://abc")
    client = make_client(mocker, "not json")
    mocker.patch("gemini_media_analyzer.make_client", return_value=client)
    mocker.patch("gemini_media_analyzer.upload_media", return_value=active)

    result = gma.analyze_media(str(media), as_json=True)

    assert result["parse_error"] == "model response was not valid JSON object"
    assert result["validation_error"] is None
    assert result["raw_text"] == "not json"


def test_analyze_media_writes_output_file(mocker, tmp_path, monkeypatch):
    media = tmp_path / "clip.mp4"
    output = tmp_path / "analysis.json"
    media.write_bytes(b"fake")
    monkeypatch.setenv("GOOGLE_API_KEY", "key")
    active = SimpleNamespace(name="files/abc", uri="gemini://abc")
    client = make_client(mocker, make_valid_response_text())
    mocker.patch("gemini_media_analyzer.make_client", return_value=client)
    mocker.patch("gemini_media_analyzer.upload_media", return_value=active)

    result = gma.analyze_media(str(media), output_path=str(output))

    saved = json.loads(output.read_text(encoding="utf-8"))
    assert saved == result
    assert saved["audio"]["transcript"][0]["text"] == "Hello"


def test_analyze_media_rejects_invalid_timeout_before_upload(
    mocker, tmp_path, monkeypatch
):
    media = tmp_path / "clip.mp4"
    media.write_bytes(b"fake")
    monkeypatch.setenv("GOOGLE_API_KEY", "key")
    mock_upload = mocker.patch("gemini_media_analyzer.upload_media")

    with pytest.raises(gma.MediaValidationError, match="timeout"):
        gma.analyze_media(str(media), timeout_seconds=0)

    mock_upload.assert_not_called()
