from __future__ import annotations

import json
from pathlib import Path

from roadsign_api.main import app
from roadsign_assist.paths import PROJECT_ROOT

DEFAULT_OUTPUT = PROJECT_ROOT / "outputs" / "api" / "openapi.json"


def export_openapi(output: Path = DEFAULT_OUTPUT) -> Path:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(app.openapi(), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return output


if __name__ == "__main__":
    print(export_openapi())
