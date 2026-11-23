# Local experiment preflight

14 November 2026

The official base-model comparison ran four outputs correctly, but its frozen plan inherited `planned_outputs: 16`. That is a real metadata defect. The original evidence remains unchanged. The new read-only validator rejects the contradiction before a future run.

Run from the project directory with the existing Python environment:

```sh
python -m renderello.experiment_preflight path/to/plan.json --root path/to/input_directory
```

This validator supports the local cases-times-seeds, native-padding plan format. It does not validate every notebook or replace provider-specific checks. It is an explicit preflight command, not automatically inserted into old notebooks.

Checks include exact planned count, bounded unique seeds, unique case identities, exact source-file hashes, actual decoded image size, source/case canvas and translation consistency, bounded native placement, and duplicate logical cells. Source paths must stay inside the evidence directory without symlinks. EXIF rotation must be normalized before freezing.

Requested template aspect and actual canvas ratio are compared by integer cross multiplication. A source-shaped canvas remains allowed, but an unmet requested aspect is explicit. Unused source-provenance entries are listed separately and are not counted as executed sources.

A passing report does not authorize generation, spending or deployment. It does not establish source ownership, physical-photo provenance, image quality or successful execution. It does not change source files, repair old evidence, or approve template ratios that do not match.

Verification: 51 new focused checks plus 94 existing matched-trial/brief-acceptance checks passed together, 145 total. A full-project regression was not rerun after this isolated module was added. The real frozen four-output plan is rejected for the stale count. A count-only correction in memory passes source/geometry checks while explicitly flagging both requested square ratios as unfulfilled and two provenance rows as unused. No corrected experiment was executed.
