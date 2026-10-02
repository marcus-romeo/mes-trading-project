"""Read-only SHA-256 verification for the separately transferred MES artifacts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RECORD_PATH = PROJECT_ROOT / "ARTIFACT_PROVENANCE_V1.json"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_artifacts(record_path: Path = RECORD_PATH) -> dict:
    """Raise on any missing, resized, or changed recorded file; write nothing."""
    record = json.loads(Path(record_path).read_text())
    if record.get("algorithm") != "SHA-256":
        raise ValueError("Unsupported artifact digest algorithm.")
    files = record.get("files", [])
    if not files:
        raise ValueError("Artifact provenance record is empty.")
    total_bytes = 0
    for item in files:
        relative = Path(item["path"])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"Unsafe artifact path in record: {relative}")
        path = PROJECT_ROOT / relative
        if not path.is_file():
            raise FileNotFoundError(path)
        size = path.stat().st_size
        if size != item["size_bytes"] or sha256_file(path) != item["sha256"]:
            raise ValueError(f"Artifact differs from recorded SHA-256: {relative}")
        total_bytes += size
    return {"passed": True, "files": len(files), "bytes": total_bytes}


if __name__ == "__main__":
    print(verify_artifacts())
