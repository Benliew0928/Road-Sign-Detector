from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

from dvc_gdrive import GDriveFileSystem

DEFAULT_FOLDERS = ("DVC Active Artifacts", "Contributor Upload Inbox")
REMOTE_NAME = "shared-gdrive"


def ensure_folder(filesystem: GDriveFileSystem, title: str) -> str:
    backend = filesystem.fs
    remote_path = f"root/{title}"
    return str(backend._get_item_id(remote_path, create=True))


def load_client_credentials(path: Path) -> tuple[str, str]:
    payload: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    client = payload.get("installed")
    if not isinstance(client, dict):
        raise ValueError(
            "Expected a Google OAuth Desktop app JSON file with an 'installed' object."
        )
    client_id = client.get("client_id")
    client_secret = client.get("client_secret")
    if not isinstance(client_id, str) or not isinstance(client_secret, str):
        raise ValueError("Desktop OAuth JSON is missing client_id or client_secret.")
    return client_id, client_secret


def configure_dvc(
    folder_id: str, client_id: str, client_secret: str, profile: str
) -> None:
    commands = (
        ("remote", "add", "-d", "-f", REMOTE_NAME, f"gdrive://{folder_id}"),
        (
            "remote",
            "modify",
            "--local",
            REMOTE_NAME,
            "gdrive_client_id",
            client_id,
        ),
        (
            "remote",
            "modify",
            "--local",
            REMOTE_NAME,
            "gdrive_client_secret",
            client_secret,
        ),
        (
            "remote",
            "modify",
            "--local",
            REMOTE_NAME,
            "profile",
            profile,
        ),
    )
    for command in commands:
        subprocess.run([sys.executable, "-m", "dvc", *command], check=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create or find the two RoadSign Assist Google Drive folders."
    )
    parser.add_argument("--profile", default="roadsign-assist")
    parser.add_argument(
        "--client-secrets-json",
        type=Path,
        help="Local Google OAuth Desktop-client JSON; never copied into the project.",
    )
    parser.add_argument(
        "--configure-dvc",
        action="store_true",
        help="Configure shared-gdrive and store its OAuth values in config.local.",
    )
    args = parser.parse_args()
    if args.client_secrets_json:
        client_id, client_secret = load_client_credentials(args.client_secrets_json.resolve())
    else:
        client_id = os.getenv("DVC_GDRIVE_CLIENT_ID")
        client_secret = os.getenv("DVC_GDRIVE_CLIENT_SECRET")
    if bool(client_id) != bool(client_secret):
        raise RuntimeError(
            "Set both DVC_GDRIVE_CLIENT_ID and DVC_GDRIVE_CLIENT_SECRET, or neither."
        )
    config = {"url": "gdrive://root", "profile": args.profile}
    if client_id and client_secret:
        config["gdrive_client_id"] = client_id
        config["gdrive_client_secret"] = client_secret
    filesystem = GDriveFileSystem(**config)
    folder_ids = {title: ensure_folder(filesystem, title) for title in DEFAULT_FOLDERS}
    if args.configure_dvc:
        if not client_id or not client_secret:
            raise RuntimeError("--configure-dvc requires custom Desktop OAuth credentials.")
        configure_dvc(
            folder_ids["DVC Active Artifacts"],
            client_id,
            client_secret,
            args.profile,
        )
    print(json.dumps(folder_ids, indent=2, sort_keys=True))
    if args.configure_dvc:
        print(
            "Configured shared-gdrive. The client credentials are local-only; "
            "the tracked config contains only the folder ID."
        )


if __name__ == "__main__":
    main()
