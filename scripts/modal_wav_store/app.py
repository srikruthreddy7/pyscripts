import datetime
import os
import re
import uuid
from typing import Optional
from urllib.parse import unquote, urlparse

import modal
from modal import App, Image, Volume, fastapi_endpoint
from fastapi import Request, HTTPException


# Modal application and dependencies
app = App("lynkup-modal-wav-store")

image = (
    Image.debian_slim()
    .pip_install("fastapi[standard]", "requests>=2.31.0")
)

# Create or access a persistent Modal Volume for audio storage
VOLUME_NAME = "lynkup-audio-volume"
audio_volume = Volume.from_name(VOLUME_NAME, create_if_missing=True)


def _sanitize_component(component: str) -> str:
    """Return a filesystem-safe component for paths."""
    safe = re.sub(r"[^a-zA-Z0-9_.-]+", "-", component.strip())
    return safe or "untitled"


def _derive_filename(audio_url: str, session_id: Optional[str]) -> str:
    parsed = urlparse(audio_url)
    base_name = os.path.basename(unquote(parsed.path or ""))

    if base_name and "." in base_name:
        name, ext = os.path.splitext(base_name)
    else:
        name, ext = ("audio", "")

    # Force .wav extension for consistency
    ext = ".wav"

    date = datetime.datetime.utcnow()
    date_prefix = f"{date.year:04d}/{date.month:02d}/{date.day:02d}"

    safe_session = _sanitize_component(session_id) if session_id else "nosession"
    safe_name = _sanitize_component(name)
    unique = uuid.uuid4().hex[:10]

    return f"{date_prefix}/{safe_session}-{safe_name}-{unique}{ext}"


def _probe_audio_format(file_path: str) -> str:
    """Return a short format label by inspecting file magic bytes.

    - "flac" if the file starts with fLaC
    - "wav" if it starts with RIFF
    - "ogg" if it starts with OggS (not treated as FLAC here)
    - "unknown" otherwise
    """
    try:
        with open(file_path, "rb") as fh:
            header = fh.read(4)
        if header.startswith(b"fLaC"):
            return "flac"
        if header.startswith(b"RIFF"):
            return "wav"
        if header.startswith(b"OggS"):
            return "ogg"
        return "unknown"
    except Exception:
        return "unknown"


@app.function(
    image=image,
    volumes={"/vol": audio_volume},
    timeout=30 * 60,
)
def fix_flac_extensions(base_dir: str = "/vol", dry_run: bool = False) -> dict:
    """Scan `base_dir` for .wav files that are actually FLAC and rename to .flac.

    Returns a summary dict with counts and a sample of renamed files.
    """
    print(f"Starting scan of {base_dir} (dry_run={dry_run})")
    renamed_pairs = []
    skipped = 0
    errors = []

    for dirpath, _, filenames in os.walk(base_dir):
        print(f"Scanning directory: {dirpath} ({len(filenames)} files)")
        for filename in filenames:
            if not filename.lower().endswith(".wav"):
                continue

            src_path = os.path.join(dirpath, filename)
            fmt = _probe_audio_format(src_path)
            print(f"  Checking {filename}: detected format = {fmt}")

            if fmt != "flac":
                skipped += 1
                continue

            dst_path = src_path[: -len(".wav")] + ".flac"
            try:
                if dry_run:
                    print(f"  [DRY RUN] Would rename: {src_path} -> {dst_path}")
                    renamed_pairs.append((src_path, dst_path))
                    continue

                if os.path.exists(dst_path):
                    # Target already exists; skip to avoid overwrite
                    print(f"  Skipping {src_path}: target {dst_path} already exists")
                    skipped += 1
                    continue

                print(f"  Renaming: {src_path} -> {dst_path}")
                os.rename(src_path, dst_path)
                renamed_pairs.append((src_path, dst_path))
            except Exception as e:
                print(f"  Error processing {src_path}: {e}")
                errors.append({"path": src_path, "error": str(e)})

    # Persist changes to the Volume
    if not dry_run:
        print("Committing changes to volume...")
        try:
            audio_volume.commit()
            print("Volume commit successful")
        except Exception as e:
            print(f"Volume commit failed: {e}")
            errors.append({"path": base_dir, "error": f"commit_failed: {e}"})

    result = {
        "renamed_count": len(renamed_pairs),
        "skipped_count": skipped,
        "errors_count": len(errors),
        "renamed_sample": renamed_pairs[:50],
        "dry_run": dry_run,
    }
    print(f"Scan complete: {result}")
    return result


@app.local_entrypoint()
def main(dry_run: bool = False, base_dir: str = "/vol"):
    """Run from CLI: `modal run scripts/modal_wav_store/app.py --dry-run True`"""
    result = fix_flac_extensions.remote(base_dir, dry_run)
    print(result)

@app.function(
    image=image,
    volumes={"/vol": audio_volume},
    timeout=15 * 60,
    secrets=[modal.Secret.from_name("functionkey")],
)
def download_to_volume(audio_url: str, session_id: Optional[str]) -> str:
    """Background job: download audio to the Modal volume and return relative path."""
    rel_path = _derive_filename(str(audio_url), str(session_id) if session_id else None)
    local_path = f"/vol/{rel_path}"
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    import requests
    with requests.get(str(audio_url), stream=True, timeout=60) as resp:
        resp.raise_for_status()
        with open(local_path, "wb") as f:
            for chunk in resp.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)
    return rel_path


@app.function(image=image, secrets=[modal.Secret.from_name("functionkey")])
@fastapi_endpoint(method="POST")
def transcribeAndAnalyze(request: Request, body: dict):
    """
    Mirror LinkProServerApi semantics:
    - Require header X-CUSTOM-API-KEY matching env VIDEO_API_SERVER_HEADER_KEY
    - Accept JSON { audioUrl: string, sessionId?: string }
    - Respond 200 { accepted: true }
    - Start background download into Modal Volume
    """
    incoming_key = request.headers.get("x-custom-api-key") or request.headers.get("X-CUSTOM-API-KEY")
    expected_key = os.environ.get("VIDEO_API_SERVER_HEADER_KEY")
    if not expected_key or not incoming_key or incoming_key != expected_key:
        raise HTTPException(status_code=401, detail="Unauthorized")

    if not isinstance(body, dict):
        raise HTTPException(status_code=400, detail="Invalid JSON")

    audio_url = body.get("audioUrl") or body.get("audio_url")
    session_id = body.get("sessionId") or body.get("session_id")
    if not audio_url or not isinstance(audio_url, str):
        raise HTTPException(status_code=400, detail="Invalid audioUrl")

    # Kick off background job to store the WAV on Modal Volume
    try:
        download_to_volume.spawn(str(audio_url), str(session_id) if session_id else None)
    except Exception:
        raise HTTPException(status_code=500, detail="Server error")

    return {"accepted": True}


@app.function(image=image)
@fastapi_endpoint(method="GET")
def healthz() -> dict:
    return {"status": "ok"}
