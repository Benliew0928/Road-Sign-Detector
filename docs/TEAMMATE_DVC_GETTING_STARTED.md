# Teammate guide: clone and restore the project with DVC

Use this guide once on a new computer. It assumes you have never used DVC.

> **What DVC does:** Git downloads the code and small `.dvc` pointer files.
> `dvc pull` uses those pointers to restore the large prepared datasets, selected
> models, OCR files, and offline audio from the shared Google Drive folder.

## 0. Get access before cloning

Send the maintainer the exact Google account email that you will use for the
project. Wait for confirmation that both of these are complete:

1. Your email was added at **Google Auth Platform → Audience → Test users**.
2. You have access to both Drive folders:
   - **DVC Active Artifacts** — Viewer is enough to pull; Editor is needed only
     if a maintainer has explicitly asked you to push an approved release.
   - **Contributor Upload Inbox** — Editor, so you can upload new candidate
     image batches.

Also obtain the project Desktop OAuth client JSON from the maintainer through a
private team-approved channel. Save it **outside** the repository, for example:

```text
C:\Users\<your-name>\Downloads\RoadSign-DVC-client.json
```

Do not commit, upload, or paste that JSON into GitHub, the DVC artifact folder,
or public chat.

## 1. Install prerequisites

You need:

- Git for Windows
- Python **3.11**
- Node.js LTS (includes `npm`)
- At least 15 GB of free disk space for the current DVC cache and working files

Open PowerShell and check the installation:

```powershell
git --version
python --version
node --version
npm --version
```

`python --version` must show Python 3.11. If the next step says that `uv` is
missing, install it once:

```powershell
python -m pip install --user uv
```

Close and reopen PowerShell afterwards if Windows does not immediately find it.

## 2. Clone the clean GitHub history

Everyone must make a fresh clone after the 2026-08-12 history rewrite.

```powershell
cd C:\Projects
git clone https://github.com/Benliew0928/Road-Sign-Detector.git
cd Road-Sign-Detector
```

Do not copy somebody else's old project folder or use an old clone.

## 3. Install the project dependencies

From the cloned `Road-Sign-Detector` folder, run:

```powershell
.\scripts\setup.ps1
```

If PowerShell blocks local scripts for this session only, run this once and then
repeat the setup command:

```powershell
Set-ExecutionPolicy -Scope Process Bypass
```

The setup creates `.venv`, installs Python packages including DVC, installs the
web dependencies, and checks the local application environment.

## 4. Configure your local Google Drive access

Replace the path below with the location where you saved the Desktop OAuth JSON.
These commands write only to ignored `.dvc/config.local` on **your** computer.

```powershell
$oauth = Get-Content -Raw "C:\Users\<your-name>\Downloads\RoadSign-DVC-client.json" | ConvertFrom-Json

.\.venv\Scripts\dvc.exe remote modify --local shared-gdrive gdrive_client_id $oauth.installed.client_id
.\.venv\Scripts\dvc.exe remote modify --local shared-gdrive gdrive_client_secret $oauth.installed.client_secret
.\.venv\Scripts\dvc.exe remote modify --local shared-gdrive profile roadsign-assist
```

Do **not** run `scripts/setup_gdrive_folders.py` as a teammate. The shared
folders and remote URL already exist in the Git clone.

## 5. Pull the DVC artifacts

Run:

```powershell
.\.venv\Scripts\dvc.exe pull
```

On the first pull, a browser window opens:

1. Choose the Google account that the maintainer added as a test user.
2. If Google says **"Google hasn't verified this app"**, choose **Continue**.
   This is the team’s testing-mode OAuth app, not a public app.
3. Review the Drive permission and choose **Continue**.
4. Return to PowerShell and wait for `dvc pull` to finish.

The first pull downloads about 5.5 GiB and can take a while. Later pulls only
download changed artifacts.

After it completes, DVC reconstructs normal project folders such as:

```text
data\processed\emtd_detection\
data\processed\emtd_segmentation\
data\processed\emtd_classification\
data\processed\classifier_no_controlled_variants_20260812\
models\exported\experimental\
models\ocr\
apps\web\public\audio\
```

You do **not** download image files manually from Drive. Folder names such as
`files/md5/0a/...` in Drive are DVC's internal hash store; do not edit them.

## 6. Verify and run the application

```powershell
.\.venv\Scripts\dvc.exe status
.\.venv\Scripts\roadsign-assist.exe doctor
.\scripts\run.ps1
```

`dvc status` should not report missing local artifacts. The last command starts
the application using the selected detector, legacy classifier baseline, and
OCR bundle.

## Daily workflow

Before starting work, update both Git and DVC:

```powershell
git pull --ff-only
.\.venv\Scripts\dvc.exe pull
```

For ordinary code work, commit and push only Git-tracked code and documentation.
Do not run `dvc push` unless the maintainer has asked you to publish an approved
new data or model release.

## When you collect new images

1. Create one uniquely named batch, for example:

   ```text
   2026-08-20_alice_kl-night-01/
     images/
     contributor_batch_manifest.csv
   ```

2. Put these columns in `contributor_batch_manifest.csv`:

   ```text
   batch_id,relative_path,source_url,collector,captured_at,location_granularity,licence_or_consent,proposed_class,sha256,notes
   ```

3. Upload that whole batch to **Contributor Upload Inbox**.
4. Tell the maintainer that the batch is ready for review.

Never drag data into **DVC Active Artifacts**. A maintainer reviews provenance,
duplicates, image quality, labels, and class gaps before accepted data becomes a
new versioned DVC release.

## Common problems

| Problem | Fix |
| --- | --- |
| `access_denied` or you cannot continue with Google | Confirm that you used the exact Google account registered as an OAuth test user. |
| Drive says you do not have access | Ask the maintainer to share the required Drive folder with that same Google account. |
| `dvc pull` cannot authenticate | Repeat step 4 and make sure the JSON file is outside the repository. |
| `dvc pull` needs too much disk space | Free disk space first; do not manually delete files under `.dvc/cache`. |
| Git shows OAuth JSON or `.dvc/config.local` | Move the JSON outside the repository and do not commit the local config. |

## Safety rules

- Never manually change `DVC Active Artifacts/files/md5/...` in Drive.
- Never commit model files, datasets, OAuth JSON, tokens, or `.dvc/config.local`.
- Never upload raw candidate batches directly to the DVC artifact folder.
- Never overwrite an existing frozen dataset release; create a new versioned
  release after review.
