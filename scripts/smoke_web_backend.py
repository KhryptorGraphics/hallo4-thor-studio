#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

from fastapi.testclient import TestClient

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from studio.backend.hallo4_studio.app import app


def main() -> int:
    client = TestClient(app)
    health = client.get("/api/health")
    assert health.status_code == 200, health.text
    runtime = client.get("/api/runtime")
    assert runtime.status_code == 200, runtime.text
    payload = runtime.json()
    assert "torch" in payload
    assert "models" in payload
    print("web backend smoke ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
