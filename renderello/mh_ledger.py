"""Durable Magic Hour spend ledger backed by a private, isolated GitLab
project (Option B).

Why this exists: the on-host JSONL ledger lives on ephemeral storage and
resets on every redeploy, and the balance anchor alone cannot see partial
top-ups. The durable ledger stores the cumulative committed spend as a single
JSON document in a ledger-only GitLab project with NO deployment hook, updated
with strict compare-and-swap through the Repository Files API:

  1. GET the file -> current content + last_commit_id.
  2. PUT new content naming that last_commit_id (optimistic concurrency).
  3. On 409 conflict, re-read and retry (bounded); any other failure raises.

Hard rules:
  - Reserve BEFORE any paid call; a failed or unreachable reserve fails
    CLOSED (the paid call never happens).
  - Reconcile moves the reservation TO the observed charge: down when the
    provider charged less, UP when it charged more (the listed price is a
    lowest-resolution floor, not a ceiling - an overcharge is real spend
    the durable ledger must record). A failed reconcile keeps the
    reservation; the balance anchor then catches any real overcharge and
    freezes spend until the owner reconciles the counters.
  - The token is a minimum project-scoped, expiring access token supplied
    via the server environment only. It is never logged, never included in
    error messages, and never written to disk by this module.
  - The ledger stores aggregate counters only: no user data, no prompts,
    no image references, no account identifiers (no PII).

Configuration (all server-side env):
  RENDERELLO_MH_DURABLE_LEDGER   "1"/"on" to enable; when enabled, ANY
                                 reserve/reconcile failure refuses the spend.
  RENDERELLO_MH_LEDGER_PROJECT   URL-encoded project path, e.g.
                                 "your-group%2Fyour-ledger-project"
  RENDERELLO_MH_LEDGER_TOKEN     project access token (api scope, <=30 days)
  RENDERELLO_MH_LEDGER_FILE      ledger path in repo (default "spend.json")
  RENDERELLO_MH_LEDGER_BRANCH    branch (default "main")
  RENDERELLO_MH_LEDGER_API       API base (default "https://gitlab.com/api/v4")
"""
from __future__ import annotations

import base64
import json
import os
import time
from urllib.parse import quote

DEFAULT_FILE = "spend.json"
DEFAULT_BRANCH = "main"
DEFAULT_API = "https://gitlab.com/api/v4"
MAX_CAS_RETRIES = 6


class LedgerError(RuntimeError):
    """Any durable-ledger failure. Callers treat this as fail-closed: the
    paid call must not proceed. Messages never carry the token or response
    bodies."""


def enabled() -> bool:
    return os.environ.get("RENDERELLO_MH_DURABLE_LEDGER", "").strip().lower() \
        in ("1", "on", "true", "yes")


def _config() -> dict:
    project = os.environ.get("RENDERELLO_MH_LEDGER_PROJECT", "").strip()
    token = os.environ.get("RENDERELLO_MH_LEDGER_TOKEN", "").strip()
    if not project or not token:
        raise LedgerError(
            "durable ledger enabled but RENDERELLO_MH_LEDGER_PROJECT / "
            "RENDERELLO_MH_LEDGER_TOKEN unset; live spend refused, zero "
            "credits billed.")
    return {
        "api": os.environ.get("RENDERELLO_MH_LEDGER_API", DEFAULT_API).rstrip("/"),
        "project": quote(project, safe="") if "/" in project else project,
        "token": token,
        "file": os.environ.get("RENDERELLO_MH_LEDGER_FILE", DEFAULT_FILE),
        "branch": os.environ.get("RENDERELLO_MH_LEDGER_BRANCH", DEFAULT_BRANCH),
    }


def _request(method: str, url: str, token: str, payload: dict | None = None):
    """One GitLab API call. Returns (status, parsed-json-or-None). Never
    raises on HTTP errors; raises LedgerError on transport/parse failure.
    Error paths do not propagate response bodies (could echo request data)."""
    import requests
    headers = {"PRIVATE-TOKEN": token}
    try:
        resp = requests.request(method, url, headers=headers, json=payload,
                                timeout=20)
    except Exception as e:
        raise LedgerError(
            f"durable ledger unreachable ({type(e).__name__}); live spend "
            "refused, zero credits billed.")
    body = None
    if resp.content:
        try:
            body = resp.json()
        except ValueError:
            body = None
    return resp.status_code, body


def _file_url(cfg: dict) -> str:
    return (f"{cfg['api']}/projects/{cfg['project']}/repository/files/"
            f"{quote(cfg['file'], safe='')}")


def read_state(cfg: dict | None = None) -> dict:
    """Current durable state: {"spent": int, "updated": str, "commit": str}.
    Raises LedgerError on any failure (fail closed)."""
    cfg = cfg or _config()
    status, body = _request(
        "GET", f"{_file_url(cfg)}?ref={quote(cfg['branch'], safe='')}",
        cfg["token"])
    if status == 404:
        raise LedgerError(
            "durable ledger file missing in the ledger project; live spend "
            "refused, zero credits billed. Seed it before enabling.")
    if status != 200 or not isinstance(body, dict):
        raise LedgerError(
            f"durable ledger read failed (HTTP {status}); live spend "
            "refused, zero credits billed.")
    try:
        content = base64.b64decode(body["content"]).decode("utf-8")
        state = json.loads(content)
        spent = int(state["spent"])
        commit = str(body["last_commit_id"])
    except (KeyError, ValueError, TypeError, json.JSONDecodeError):
        raise LedgerError(
            "durable ledger content is malformed; live spend refused, zero "
            "credits billed.")
    return {"spent": spent, "updated": str(state.get("updated", "")),
            "commit": commit}


def _write_state(cfg: dict, spent: int, last_commit_id: str) -> None:
    """CAS write: update the ledger only if it is still at last_commit_id.
    Raises LedgerError on conflict (caller retries) or any failure."""
    payload = {
        "branch": cfg["branch"],
        "content": json.dumps({
            "spent": int(spent),
            "updated": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }, indent=2) + "\n",
        "commit_message": f"ledger: spent -> {int(spent)}",
        "last_commit_id": last_commit_id,
    }
    status, _ = _request("PUT", _file_url(cfg), cfg["token"], payload)
    if status == 200:
        return
    if status in (400, 409):
        # 400 is GitLab's "file changed" validation failure on a stale
        # last_commit_id; 409 is the explicit conflict. Both mean: retry.
        raise LedgerError("cas-conflict")
    raise LedgerError(
        f"durable ledger write failed (HTTP {status}); live spend refused, "
        "zero credits billed.")


def _mutate(delta: int) -> int:
    """Apply `delta` to the durable spent counter with CAS retries. Returns
    the new committed total. Raises LedgerError on persistent failure."""
    import random
    cfg = _config()
    for attempt in range(MAX_CAS_RETRIES):
        state = read_state(cfg)
        new_spent = state["spent"] + delta
        if new_spent < 0:
            new_spent = 0
        try:
            # The commit id sent here always comes from this client's own
            # read, so it always names a REAL commit. GitLab rejects a
            # stale real id with HTTP 400 (verified live 2026-11-20),
            # which is what makes this CAS exactly-one-wins. GitLab does
            # NOT reject a nonexistent commit id (live-verified gap), so
            # never construct one: only ids returned by read_state.
            _write_state(cfg, new_spent, state["commit"])
            return new_spent
        except LedgerError as e:
            if str(e) == "cas-conflict" and attempt < MAX_CAS_RETRIES - 1:
                time.sleep(0.2 * (attempt + 1) + random.uniform(0, 0.3))
                continue
            if str(e) == "cas-conflict":
                raise LedgerError(
                    "durable ledger update kept conflicting; live spend "
                    "refused, zero credits billed.")
            raise
    raise LedgerError("durable ledger update failed; refused.")  # unreachable


def reserve(credits: int) -> int:
    """Reserve `credits` BEFORE the paid call. Returns the new durable
    total. Any failure raises LedgerError -> caller must refuse the spend
    (fail closed). No-op caller responsibility when not enabled()."""
    if credits <= 0:
        raise LedgerError("reserve amount must be positive")
    return _mutate(int(credits))


def reconcile(reserved: int, observed: int) -> int:
    """After the paid call, move the reservation TO the observed charge.
    Downward when the observed charge is lower; UPWARD when the provider
    charged more than the listed reservation - the overcharge is real
    spend and the durable ledger must record it, otherwise a restart
    (which drops the local reservation) would permanently under-count.
    Unknown observed charges must NOT reach this function: callers keep
    the reservation instead. Raises LedgerError on failure - callers keep
    the reservation; the balance anchor then catches the discrepancy and
    freezes spend until the owner reconciles the counters."""
    delta = int(observed) - int(reserved)
    if delta == 0:
        cfg = _config()
        return read_state(cfg)["spent"]
    return _mutate(delta)


def spent() -> int:
    """Durable committed spend. Raises LedgerError on failure."""
    return read_state()["spent"]
