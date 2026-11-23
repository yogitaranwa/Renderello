"""Build the single self-contained Renderello Colab notebook.

The notebook vendors the whole renderello package source inline (one cell
writes every module), so it runs on a fresh Colab runtime with no repo
checkout and no Drive mount requirement. It exposes the SAME studio pipeline
as the website: multi-product + reference inputs, 12 presets, long detailed
prompts, procedural / SDXL-Lightning(GPU) / Magic Hour providers.

Colab free-tier ToS (verified 2026-11-20 against the official FAQ): free
runtimes disallow remote proxies and serving web UIs for content
generation. This notebook is INTERACTIVE ONLY - uploads and results happen
through Colab's own UI (files.upload / widgets), never through a tunnel.
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PKG = ROOT / "renderello"


def md(source: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": source.splitlines(keepends=True)}


def code(source: str) -> dict:
    return {"cell_type": "code", "execution_count": None, "metadata": {},
            "outputs": [], "source": source.splitlines(keepends=True)}


def vendor_cells() -> list:
    """One visible %%writefile cell per module - the full implementation is
    embedded in the notebook and readable, not a downloaded blob."""
    cells = []
    files = sorted(PKG.rglob("*.py"))
    header = (
        "import pathlib, sys\n"
        "base = pathlib.Path('/content/renderello-src')\n"
        "base.mkdir(parents=True, exist_ok=True)\n"
        "sys.path.insert(0, str(base))\n"
        "print('writing', " + str(len(files)) + ", 'embedded modules into', base)"
    )
    cells.append(code(header))
    seen_dirs = set()
    for f in files:
        rel = f.relative_to(ROOT).as_posix()
        d = str(Path(rel).parent)
        if d not in seen_dirs:
            seen_dirs.add(d)
            cells.append(md(f"### embedded source: `{d}/`"))
        text = f.read_text()
        body = f"# ---- embedded module: {rel} ----\n"
        body += f"path = base / {rel!r}\n"
        body += "path.parent.mkdir(parents=True, exist_ok=True)\n"
        body += f"path.write_text(SRC)\nprint('wrote', path)\n"
        cell_src = "SRC = " + repr(text) + "\n" + body
        cells.append(code(cell_src))
    cells.append(code("import importlib, sys\n"
                      "sys.path.insert(0, '/content/renderello-src')\n"
                      "import renderello\nimportlib.reload(renderello)\n"
                      "print('renderello package ready:', renderello.__file__)"))
    return cells


CELLS = [
    md("# Renderello - AI Product Photography Studio\n"
       "One self-contained notebook: multi-image product + reference inputs, "
       "12 researched shoot presets, very detailed prompts, and three engines "
       "(procedural CPU, SDXL-Lightning on this GPU, Magic Hour API).\n\n"
       "**How to use:** Runtime -> Run all, then follow the sections in order. "
       "Interactive use only - Colab's free runtime does not allow serving "
       "content-generation web UIs, so everything runs through this notebook UI."),
    md("## 0. Install dependencies"),
    code("!pip -q install fastapi uvicorn python-multipart 'pillow>=12.3,<13' numpy rembg onnxruntime kornia diffusers transformers accelerate safetensors huggingface_hub\n"
         "print('installs done')"),
    md("## 1. The full Renderello pipeline, embedded (same code as the website)\n"
       "Every module of the project is written out below by one visible cell each - "
       "nothing is downloaded as project source. Only pip packages and (for the learned "
       "cutout) the public BiRefNet model weights download at runtime."),
    code("__VENDOR__"),
    md("## 2. Choose an engine\n"
       "- **procedural**: free, CPU, deterministic backdrop; product pixels stay source-exact.\n"
       "- **sdxl_lightning**: free AI scene on this Colab GPU; the product is composited on top.\n"
       "- **magichour**: Magic Hour AI editor. In this notebook the local engine runs as a dry-run "
       "mock: it shows the request that would be sent and spends nothing. LIVE calls never run from "
       "the notebook's local engine by design: a notebook-side ledger could not see website spend and "
       "the two together could exceed the authorized total. Live runs go through the hosted website "
       "bridge in section 7, so one server-side ledger, balance anchor and budget guard protect every "
       "live surface. The Magic Hour integration itself is unchanged and remains in the pipeline code "
       "below; the procedural and SDXL engines are fully functional. Live calls use one input image "
       "per call, seedream-v5-flash at 640px, documented normal price 30 credits per image (the "
       "provider's documented list price, not a billing guarantee).\n\n"
       "Spend guard (server-side, applies to every live surface): a per-call reservation against the "
       "authorized credit budget, an account-balance anchor, and a durable ledger. If the budget, "
       "anchor, or ledger configuration is missing or unreadable, the guard refuses BEFORE any paid "
       "call and zero credits are billed. The deployment's ledger credentials and API key are private "
       "to its owner, are never shared, and must never be pasted into this notebook.\n\n"
       "Cutouts use the best learned backend available: BiRefNet when torch and kornia are importable, "
       "otherwise the compact U2Net-p ONNX model, otherwise an OpenCV grabcut fallback (cruder edges). "
       "The backend actually used is printed in section 6 and recorded in manifest.json."),
    code("provider = 'procedural'  # 'procedural' (free) | 'sdxl_lightning' (free, GPU) | 'magichour' (dry-run mock)\n"
         "mhModel = 'seedream-v5-flash'  # live allowlist default: one input image, 640px, 30 credits\n"
         "liveMode = False          # dry-run mock; live calls are blocked in this notebook (see above)\n"
         "creditBudget = 30         # maximum credits a live run may spend; above this it is refused\n"
         "import os\n"
         "os.environ['RENDERELLO_MH_BUDGET_CREDITS'] = str(creditBudget)\n"
         "if provider == 'magichour' and liveMode:\n"
         "    raise RuntimeError(\n"
         "        'Live Magic Hour calls never run from the notebook local engine: they go through '\n"
         "        'the hosted website bridge in section 7, so one server-side ledger guards every '\n"
         "        'live surface. Use liveMode = False here (dry-run mock) or section 7 for live runs.')\n"
         "print('engine:', provider, '| magichour model:', mhModel, '| live:', liveMode)"),
    md("## 3. Upload product photo(s) and optional reference image(s)\n"
       "Every product photo is kept pixel-exact in compose mode; reference images steer style only, never composited."),
    code("from google.colab import files\n"
         "import pathlib\n"
         "up = pathlib.Path('/content/uploads'); up.mkdir(exist_ok=True)\n"
         "def _save(uploaded):\n"
         "    saved = []\n"
         "    for name, data in uploaded.items():\n"
         "        p = up / pathlib.Path(name).name\n"
         "        p.write_bytes(data)\n"
         "        saved.append(p)\n"
         "    return saved\n"
         "print('Upload PRODUCT photo(s):')\n"
         "products = _save(files.upload())\n"
         "print('Upload REFERENCE image(s) (optional - cancel to skip):')\n"
         "try:\n"
         "    refs = _save(files.upload())\n"
         "except Exception:\n"
         "    refs = []\n"
         "print('products:', [p.name for p in products], '| references:', [p.name for p in refs])"),
    md("## 4. Pick a preset and describe the shot"),
    code("from renderello.presets import PRESETS\n"
         "for key, preset in PRESETS.items():\n"
         "    print(f'{key:18s} {preset.name:22s} {preset.tagline}')\n"
         "presetKey = 'clean_studio'\n"
         "productDescription = ''   # e.g. '100 ml amber glass fragrance bottle with black cap'\n"
         "materialChoice = ''       # matte_plastic|glossy_plastic|glass|metal|fabric|ceramic|wood|leather (blank = auto)\n"
         "instructionText = ''      # scene, mood, props, palette, camera; environment only, never the product\n"
         "variantCount = 2"),
    md("## 5. Generate"),
    code("from renderello.studio import StudioRequest, run_studio\n"
         "runResult = run_studio(StudioRequest(\n"
         "    product_paths=products, preset_key=presetKey, instruction=instructionText,\n"
         "    reference_paths=refs, variants=variantCount, provider=provider,\n"
         "    material=materialChoice or None, product_description=productDescription,\n"
         "    knob_values={'mh_model': mhModel}), pathlib.Path('/content/runs'))\n"
         "print('run dir:', runResult.run_dir)\n"
         "for variant in runResult.variants:\n"
         "    print('variant:', variant.image or f'FAILED: {variant.error}')"),
    md("## 6. Review and download\n"
       "Contact sheet + manifest (prompt, QA verdicts, credits) included. "
       "QA honesty: whole-image AI edits regenerate the product - verify label/logo fidelity before publishing."),
    code("from PIL import Image\n"
         "import json\n"
         "display(Image.open(runResult.run_dir / 'contact_sheet.png'))\n"
         "manifest = json.loads((runResult.run_dir / 'manifest.json').read_text())\n"
         "print('mode:', manifest['mode'], '| provider:', manifest['provider_used'])\n"
         "for p in manifest.get('products', []):\n"
         "    print('product:', p['file'], '| cutout backend:', p.get('cutout_backend', 'n/a'),\n"
         "          '| coverage:', p.get('cutout_coverage', 'n/a'))\n"
         "for w in manifest.get('warnings', []):\n"
         "    print('WARNING:', w)\n"
         "for v in manifest['variants']:\n"
         "    print(v['image'], '| qa:', (v.get('qa') or {}).get('publish_ready'), '|', '; '.join(v.get('warnings') or []))\n"
         "import shutil\n"
         "shutil.make_archive('/content/renderello_run', 'zip', runResult.run_dir)\n"
         "files.download('/content/renderello_run.zip')"),
    md("## 7. (Optional) One live Magic Hour run through the hosted website\n"
       "Live spend goes through the hosted website's guarded endpoints, so the SAME server-side "
       "budget guard, balance anchor and durable spend ledger protect this notebook and the "
       "website - one ledger, no double counting.\n\n"
       "- Sign in with YOUR OWN website account; it is entered at runtime and is never stored "
       "in this notebook.\n"
       "- No Magic Hour API key and no ledger token appear in this notebook; they stay "
       "server-side.\n"
       "- Nothing is spent automatically: step 7.2 prints the server's own estimate, step 7.3 "
       "needs your typed confirmation, and the server reserves and checks the credits before any "
       "paid call - a refusal bills zero credits.\n"
       "- Exactly one paid image per confirmed run: the first uploaded product photo, one "
       "template, one variant (documented normal price 30 credits; the provider's list price, "
       "not a billing guarantee).\n"
       "- Live runs work only while the deployment owner has Magic Hour enabled on the server; "
       "otherwise the server refuses before any paid call.\n"
       "- The session lives only in this runtime's memory; step 7.5 signs out."),
    code("# 7.1 Sign in to the hosted website (your own account, runtime only)\n"
         "bridgeEnabled = False  # explicit opt-in only; this is NOT payment approval\n"
         "typedConfirmation = None\n"
         "spendConfirmationToken = None\n"
         "bridgeQuoteToken = None\n"
         "bridgeSessionToken = None\n"
         "site = None  # discard any stale runtime session before the default-off guard\n"
         "assert bridgeEnabled is True, 'Paid bridge disabled: no credentials requested and no request sent'\n"
         "import secrets\n"
         "bridgeSessionToken = secrets.token_hex(16)\n"
         "siteUrl = 'https://renderello-studio.onrender.com'\n"
         "import json, requests, getpass\n"
         "site = requests.Session()\n"
         "signIn = site.post(siteUrl + '/api/login', allow_redirects=False, json={\n"
         "    'email': getpass.getpass('Website sign-in email: '),\n"
         "    'password': getpass.getpass('Website sign-in password: ')}, timeout=30)\n"
         "print('sign-in status:', signIn.status_code)\n"
         "signIn.raise_for_status()\n"
         "del signIn  # drop the auth response object; only the session cookie persists"),
    code("# 7.2 Server-side estimate (never spends)\n"
         "assert globals().get('bridgeEnabled') is True and globals().get('bridgeSessionToken'), 'Bridge disabled: run enabled 7.1 first'\n"
         "assert globals().get('site') is not None, 'No current bridge session'\n"
         "bridgeQuoteToken = None\n"
         "bridgePreset = 'clean_studio'\n"
         "bridgeSceneAttempts = 1  # no automatic second paid scene attempt\n"
         "estimate = site.get(siteUrl + '/api/estimate', timeout=30, params={\n"
         "    'provider': 'magichour', 'mh_model': mhModel, 'variants': 1,\n"
         "    'presets': bridgePreset, 'mh_route': 'scene',\n"
         "    'knob_scene_max_attempts': str(bridgeSceneAttempts)}).json()\n"
         "print(json.dumps(estimate, indent=2))\n"
         "assert estimate.get('credits_per_image'), 'server estimate unavailable - stop here'\n"
         "assert estimate.get('within_budget') is True, \\\n"
         "    'server says this batch exceeds the approved budget - stop, zero credits billed'\n"
         "assert estimate.get('within_batch_limit') is True, \\\n"
         "    'server says this batch exceeds the per-batch limit - stop, zero credits billed'\n"
         "bridgeQuoteToken = bridgeSessionToken"),
    code("# 7.3 Explicit confirmation - nothing is sent to Magic Hour before this cell\n"
         "assert globals().get('bridgeEnabled') is True, 'Bridge disabled'\n"
         "assert globals().get('bridgeSessionToken') and globals().get('bridgeQuoteToken') == bridgeSessionToken, 'Get a fresh estimate in 7.2'\n"
         "typedConfirmation = None\n"
         "spendConfirmationToken = None\n"
         "print('Documented normal price for this request:',\n"
         "      estimate['credits_per_image'], 'credits (list price, not a billing guarantee).')\n"
         "print('Budget state reported by the server:', estimate['budget'])\n"
         "typedConfirmation = input('Type exactly SPEND to authorize ONE live image: ')\n"
         "assert typedConfirmation == 'SPEND', 'not confirmed - nothing was sent, zero credits billed'\n"
         "spendConfirmationToken = bridgeSessionToken"),
    code("# 7.4 The single live call + download (everything paid-adjacent happens HERE,\n"
         "# while the session is alive; the finally-block signs out only after downloads)\n"
         "assert globals().get('bridgeEnabled') is True, 'Bridge disabled'\n"
         "assert globals().get('bridgeSessionToken') and globals().get('spendConfirmationToken') == bridgeSessionToken, 'No fresh spending confirmation for this session'\n"
         "spendConfirmationToken = None  # consume before any paid POST\n"
         "import time, re\n"
         "assert typedConfirmation == 'SPEND', \\\n"
         "    'stale cell: rerun 7.3 and type SPEND again - a consumed confirmation cannot spend twice'\n"
         "typedConfirmation = None  # consumed BEFORE the POST: a rerun of this cell cannot double-charge\n"
         "spentBefore = estimate['budget']['spent']\n"
         "print('pre-POST durable spent baseline:', spentBefore, \\\n"
         "      '- write this down; recovery below compares against it')\n"
         "job = {}\n"
         "savedFiles = []\n"
         "try:\n"
         "    with open(products[0], 'rb') as fh:\n"
         "        try:\n"
         "            sent = site.post(siteUrl + '/api/generate', timeout=120,\n"
         "                allow_redirects=False,\n"
         "                files={'products': (products[0].name, fh)},\n"
         "                data={'preset': bridgePreset, 'provider': 'magichour', 'variants': '1',\n"
         "                      'instruction': instructionText,\n"
         "                      'product_description': productDescription, 'mh_model': mhModel,\n"
         "                      'mh_route': 'scene', 'knob_scene_max_attempts': str(bridgeSceneAttempts)})\n"
         "        except requests.Timeout:\n"
         "            raise SystemExit(\n"
         "                'POST timed out: the server MAY have accepted the job. '\n"
         "                'DO NOT RERUN this cell and DO NOT start a new run - a rerun '\n"
         "                'would submit a second paid job. Recovery: sign in again (7.1), '\n"
         "                're-run 7.2 and compare budget.spent with the baseline printed '\n"
         "                'above. A rise by credits_per_image means the job was accepted; '\n"
         "                'an unchanged value is NOT proof it was not submitted (another '\n"
         "                'website session can move the same budget). Check the website '\n"
         "                'Results tab and NEVER resubmit until the deployment owner '\n"
         "                'confirms the job state.')\n"
         "    print('generate status:', sent.status_code)\n"
         "    if sent.status_code >= 500:\n"
         "        raise SystemExit(\n"
         "            f'server error {sent.status_code}: paid state UNKNOWN - the request '\n"
         "            'may have been accepted before the failure. DO NOT RERUN this cell. '\n"
         "            'Recover exactly as for a timeout: re-login (7.1), re-run 7.2, '\n"
         "            'compare budget.spent with the baseline, check the Results tab, and '\n"
         "            'never resubmit until the deployment owner confirms.')\n"
         "    if sent.status_code != 200:\n"
         "        print('server refusal detail:', sent.text[:400])\n"
         "    sent.raise_for_status()  # 4xx refusals bill zero credits\n"
         "    jobId = sent.json()['job_id']\n"
         "    del sent\n"
         "    deadline = time.monotonic() + 900\n"
         "    while time.monotonic() < deadline:\n"
         "        time.sleep(5)\n"
         "        job = site.get(siteUrl + '/api/jobs/' + jobId, timeout=30).json()\n"
         "        if job.get('status') != 'running':\n"
         "            break\n"
         "    if job.get('status') == 'running':\n"
         "        raise SystemExit(\n"
         "            f'job {jobId} still running after 15 minutes. It may still spend '\n"
         "            'or finish. DO NOT RERUN this cell. This cell signed out in its '\n"
         "            'finally block, so sign in again with 7.1 first, then poll '\n"
         "            'GET /api/jobs/' + jobId + ' with the new session, or check the '\n"
         "            'Results tab. Never start a new paid job until this one resolves.')\n"
         "    print('job:', job.get('status'), job.get('error') or '')\n"
         "    assert job.get('status') == 'done', (\\\n"
         "        'live run did not complete (job id ' + jobId + '). A failed or '\n"
         "        'interrupted job MAY STILL HAVE BILLED credits. DO NOT RERUN this '\n"
         "        'cell: sign in again (7.1), re-run 7.2 and check budget.spent and '\n"
         "        'the Results tab before any new attempt.')\n"
         "    runDir = job['runs'][0]['run_dir']\n"
         "    # Result paths must be server-relative run ids on the pinned host - never\n"
         "    # follow an absolute URL or unexpected shape returned inside job data.\n"
         "    assert re.fullmatch(r'[0-9]{8}-[0-9]{6}-[0-9a-f]+', runDir), \\\n"
         "        'unexpected run id shape - refusing to fetch an unverified result path'\n"
         "    for name in ['contact_sheet.png', 'manifest.json']:\n"
         "        blob = site.get(siteUrl + '/api/runs/' + runDir + '/' + name, timeout=60)\n"
         "        if blob.status_code != 200:\n"
         "            raise SystemExit(\n"
         "                f'result download failed ({blob.status_code}) for {name}. The '\n"
         "                'paid job COMPLETED - do not rerun 7.4. Sign in again (7.1) and '\n"
         "                'retry only this download block, or use the Results tab.')\n"
         "        out = pathlib.Path('/content/live_' + name)\n"
         "        out.write_bytes(blob.content)\n"
         "        savedFiles.append(str(out))\n"
         "        print('saved', out, len(blob.content), 'bytes')\n"
         "        del blob\n"
         "finally:\n"
         "    # Sign out on every path - after downloads, refusal or failure. The server\n"
         "    # drops only this runtime's own session; other browser sessions persist.\n"
         "    site.post(siteUrl + '/api/logout', allow_redirects=False, timeout=30)\n"
         "    print('signed out; live session closed')"),
    code("# 7.5 Review the downloaded result (files already saved in 7.4; no session needed)\n"
         "assert globals().get('bridgeEnabled') is True, 'Bridge disabled'\n"
         "from PIL import Image\n"
         "display(Image.open('/content/live_contact_sheet.png'))\n"
         "manifestLive = json.loads(pathlib.Path('/content/live_manifest.json').read_text())\n"
         "print(json.dumps({k: manifestLive.get(k) for k in\n"
         "      ('mode', 'provider_used', 'budget', 'knobs') if k in manifestLive}, indent=2))\n"
         "print('eye review remains the acceptance gate - QA scores are screens, not proof')"),
]


def build() -> Path:
    cells = []
    for c in CELLS:
        if c["cell_type"] == "code" and c["source"] == ["__VENDOR__"]:
            cells.extend(vendor_cells())
            continue
        cells.append(c)
    nb = {
        "nbformat": 4, "nbformat_minor": 5,
        "metadata": {"colab": {"provenance": []},
                     "kernelspec": {"name": "python3", "display_name": "Python 3"},
                     "accelerator": "GPU"},
        "cells": cells,
    }
    out = ROOT / "notebooks" / "Renderello-Studio.ipynb"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(nb, indent=1))
    return out


if __name__ == "__main__":
    p = build()
    print("wrote", p, p.stat().st_size, "bytes")
