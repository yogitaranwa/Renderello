# Rollback

How to undo a change if it breaks something.

## Website (Render)

- The service auto-deploys on push to the connected branch. Rollback is: revert to the known-good commit and push, then confirm the deploy goes live.
- Known-good reference at handover: the `ac6114d` source checkpoint (21 November 2026), which is the tree this folder ships.
- Environment variables live in the Render dashboard. If a breakage followed an env change, restore the previous value there; keep the secret values out of source and chat.
- After any rollback, smoke-check: login page loads, sign-in works, `/api/presets` answers, one procedural generation completes.

## Notebook and documents (Drive)

- Every file in this handover keeps its Drive revision history. To restore an earlier version of any document or notebook, open the file's version history in Google Drive and restore the revision.
- The state of the key documents immediately before the 21 November 2026 polish pass is also preserved as plain copies in the sibling folder `GitHub-handover-pre-polish-backup-20261121` (outside this push-ready tree).

## Generated data

- Website run directories and the logout denylist are disposable: they expire (24 h janitor, 12 h token TTL) and can be deleted without affecting the product.
- The durable spend ledger is a single JSON document in the isolated ledger project. Do not edit it by hand; corrections go through the reconcile path so the audit trail stays intact.

## Notebook specifics

- If a Colab run breaks after a dependency change, restart the runtime and Run all again; the dependency-check cell may itself restart the kernel on a fresh runtime, which is expected, not a failure.
- If the embedded package write fails mid-run, rerun the assembly cell before any generation cell.
