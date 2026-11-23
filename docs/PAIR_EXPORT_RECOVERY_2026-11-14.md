# Recoverable provider-attempt evidence export
14 November 2026, 12:00 IST. Offline reliability work, not a photographic result.

## Change
Provider-attempt export formerly wrote the economics payload followed by its provenance references. A disk or permission error between those writes could leave a lone payload. New exports stage both JSON byte streams and a hash-bound completion marker in a unique, private staging directory beside the destination. Both files are serialized before any staging write. Staged file bytes are flushed before publication. POSIX directories are also flushed. Windows uses file fsync only because its Python CRT cannot open directories for fsync; the marker explicitly records file_fsync_only_windows. No directory-entry persistence or power-loss durability is promised there. Other directory permission/sync errors still fail, rather than being swallowed. Destination names use exclusive hard links, not replacement or a check-then-overwriting write. The completion marker is published last and binds the absolute payload path and the two SHA256 values.

This is a recoverable protocol, NOT an atomic multi-file rename or a promise against arbitrary filesystem/power-loss behavior. A failed publication can leave one or two destination files. It exits with an error, names the retained staging directory and never describes that pair as complete. All partial files and staging bytes stay available. If the staging writes themselves failed, recovery may correctly refuse incomplete or corrupt staged data; fix storage and rerun the original import to a new filename, keeping the failed evidence.

New payloads include evidence_pair_required=true. The existing brief-economics CLI validates their marker, payload and sidecar before computing a report. The Python from_payload adapter requires evidence_path for these exports and verifies it against the supplied payload. Copying the payload alone therefore fails rather than silently dropping provenance. Legacy unpaired input dictionaries remain supported and are not retroactively authenticated. Removing the contract field deliberately is not authenticated evidence; this protocol is not a signature or adversarial tamper-proof storage.

## Commands

```
python -m renderello.cli provider-attempts --input import-spec.json --output attempts.json
python -m renderello.cli provider-attempts --validate-output --output attempts.json
python -m renderello.cli brief-economics --input attempts.json --output economics.json
```

On a publication failure with a complete retained staging directory:

```
python -m renderello.cli provider-attempts --recover-stage /absolute/path/.attempts.json.stage-RETURNED_SUFFIX --output /absolute/path/attempts.json
```

Use the actual path returned in the error, not the illustrative suffix. Recovery checks the staged hashes, original destination, all existing destinations and conflicts before publishing missing names. Matching files from a partial publication are kept, never overwritten. Conflicts, broken symlinks, changed stages or a different destination are refused. A concurrent destination conflict also fails without clobbering it. Repeating explicit recovery is idempotent. Successful and failed staging directories are retained, not automatically deleted.

The stage and destination must be in the same directory; a filesystem without hard-link support fails safely rather than switching to an overwriting fallback. Paths are intentionally bound. Relocation is an explicit evidence-recovery task, not an automatic rebind of the marker. Three ordinary JSON files copied without their stages can validate at their original destination; moving them to another path requires a new declared export, not editing a marker in place.

## Verification scope
102 focused tests passed across pair export, provider-attempt evidence and economics, with 30 existing Pillow warnings. Fault injection covers each staging write, actual partial writes, file fsync, directory fsync, publication of each name, pre-existing files, dangling symlinks, concurrent destination conflicts, corrupt stages/committed pairs, wrong recovery destination, serialization refusal, CLI validation/recovery and paired economics refusal/round-trip. All evidence and both focused-run fixtures/logs are preserved. No provider call, model download or photographic improvement. This is not a new full-project regression; the earlier 3,617-pass/2-skip full collection belongs to the prior source snapshot.

## Remaining substantive brief work
The original brief's 20 qualified varied real photos, eight polished template covers, two matched blind leading-specialist comparisons, seven-score/critical-error acceptance and current quality/value economics remain unresolved. Code and synthetic checks cannot supply that evidence. Paid Magic Hour calls remain paused. Existing saved native model trials can be organized and reviewed as a bounded next step, explicitly labelled assistant review rather than independent blind acceptance. They cannot manufacture the missing matched specialist exports or enough accepted templates. Any future generation requires its own permitted free route or owner authorization and verified costs. More importer fault tests are not the next quality milestone.

## Windows correction, 16:11 IST
The actual Windows run exposed PermissionError13 when opening a staging directory for POSIX fsync. The platform-specific correction keeps file fsync, exclusive hard links, marker-last publication, retained stages and hash validation. New markers declare persistence; old markers remain readable as unspecified. CLI output reports the declaration and no power-loss guarantee. Recovery retains the original declaration, never upgrading it on a different host. This fixes a real failed export, not a photographic quality result.
