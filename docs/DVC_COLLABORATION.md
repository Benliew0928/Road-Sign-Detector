# Shared DVC workflow

Git stores code, manifests, and `.dvc` pointer files. DVC stores the active
prepared datasets, selected model artifacts, and offline runtime audio. The
local `_archive/` folder is never pushed.

## One-time Google Drive setup

As of 2026-08-12, Google blocks DVC's shared built-in OAuth application. Create
a project-owned Desktop OAuth client instead:

1. In Google Cloud, enable the Google Drive API.
2. Configure the OAuth consent screen. For an external app, add every teammate
   as a test user; for a university-owned app, follow the institution's admin
   policy.
3. Create an OAuth client with application type **Desktop app**.
4. Download the Desktop client's JSON file. Keep it outside the repository,
   then create/find both folders and configure DVC in one command:

```powershell
.\.venv\Scripts\python.exe scripts\setup_gdrive_folders.py `
  --client-secrets-json "C:\path\to\client_secret.json" `
  --configure-dvc
```

5. Push the active artifacts:

```powershell
.\.venv\Scripts\dvc.exe push
```

The helper writes only the Drive folder ID to tracked `.dvc/config`. OAuth
client values go to ignored `.dvc/config.local`, and user tokens remain in the
credential cache. The downloaded JSON, local config, and tokens must never be
committed or sent through chat.

Share both Drive folders with the exact teammate Google accounts as editors.
Do not enable public/anonymous editor access.

## Teammate setup

```powershell
git clone https://github.com/Benliew0928/Road-Sign-Detector
cd Road-Sign-Detector
.\scripts\setup.ps1
$env:DVC_GDRIVE_CLIENT_ID = "team-desktop-client-id"
$env:DVC_GDRIVE_CLIENT_SECRET = "team-desktop-client-secret"
.\.venv\Scripts\dvc.exe remote modify --local shared-gdrive gdrive_client_id $env:DVC_GDRIVE_CLIENT_ID
.\.venv\Scripts\dvc.exe remote modify --local shared-gdrive gdrive_client_secret $env:DVC_GDRIVE_CLIENT_SECRET
.\.venv\Scripts\dvc.exe pull
```

Each teammate authenticates with their own Google account. Never send OAuth
tokens, credential-cache JSON files, or a service-account key through Git.

## Adding contributor data

1. Upload a uniquely named batch to **Contributor Upload Inbox**, for example
   `2026-08-20_alice_kl-night-01/`.
2. Include `contributor_batch_manifest.csv` with source, collector, captured-at,
   location granularity, consent/licence, semantic label if known, and SHA-256.
3. A maintainer downloads the batch into an isolated review directory, checks
   provenance and duplicates, and records accept/reject decisions.
4. Create a new versioned prepared release. Never modify an existing release in
   place.
5. Run the audit, `dvc add` the new release, commit the changed `.dvc` pointer
   and manifests, `dvc push`, then push Git.

The inbox is human-facing intake; the DVC folder is machine-managed storage.
Never drag files manually into **DVC Active Artifacts**.
