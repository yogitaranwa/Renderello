# Per-attempt provider evidence integration
14 November 2026, 11:23 IST. Offline implementation and mocked contract verification, not a real billing or photo-quality result.

## Gap fixed
Normal pure_prompt/hybrid variants previously wrote backend and latency but no provider project/actual charge. The old credits_estimated field defaulted to zero. The provider separately held response details and billing logs; this required manual joining and could be misread as free generation.

MagicHourBackend now resets last_result before every edit, creates a unique attempt ID and records stages before submission, after accepted ID and after decoded output. Current poll detail is retained even for error/canceled/pending statuses. Accepted project ID is stored before receipt-file writes, so a write error cannot accidentally leave a previous ID in memory. The generation POST is still never automatically retried. A timeout with no project ID is ambiguous, not permission to submit again.

Pipeline manifests now attach sanitized provider_evidence to each called editor variant, including failures after provider completion, failed downloads and failed QA/partial artifacts. Fields include attempt/provider/model/mock/project/status/stage, terminal recorded charge or null, separate submission/latest charge observations, receipt/log references, output SHA256 when present, and pipeline outcome. No auth headers, API keys, signed download URLs or raw provider payloads are copied into this record. A terminal error with an explicit charge keeps that charge. A submission or rendering-stage charge is not settled billing. Invalid, negative, boolean, nonfinite or missing charges remain unknown. Provider-supplied terminal charge is a recorded observation, not independent billing/account authentication or an assertion that no later refund can occur.

Only current call IDs are attached: a preflight failure cannot reuse stale evidence from the preceding call. Cutout-preflight failures explicitly record no provider call. Legacy credits_estimated remains for compatibility but is labelled not actual charge; do not add it to actual costs or default missing provider costs to zero. Null provider evidence for non-Magic-Hour routes does not establish zero serving costs. Local infrastructure costs remain separate.

## Main entry point to existing economics records

```
python -m renderello.cli provider-attempts --input import-spec.json --output new-attempts.json
python -m renderello.cli brief-economics --input new-attempts.json --output new-economics.json
```

The import spec has plan (existing PlannedOutput objects), as_of ISO date and ordered runs. Each run has manifest and mapping: a list of {variant_index: integer, output_id: preregistered ID}. Optional charge_credit_system is an explicit caller declaration, not read from model names. Relative manifest paths resolve against the spec. Every recorded provider variant in each supplied manifest must be mapped, including failures, so billing denominator cannot cherry-pick successful slots. The plan source SHA256 must match encoded source intake identity, and template/model must match the manifest. Methods in this adapter are model names; evaluate different route/configuration experiments in separate explicitly named plans rather than merging them as a single quality claim.

The adapter checks output bytes against the recorded hash for successful artifacts. Missing/changed files, legacy missing per-call evidence, duplicate attempt IDs and repeated provider project IDs are refused. Recovery of one billed project is not a new billed generation. List repeated attempts against the same planned output in their actual chronological order with first_try semantics declared in the plan. No retry is inferred from filenames or failure count. Unknown provider outcome becomes an unknown Attempt; terminal failed/partial execution becomes failed even when billed. Missing charges stay None. Exact repeated physical outputs are deduplicated later by the existing economics report.

Output is compatible with brief_economics.from_payload and contains blank ratings, never invented judgments or a winner. A separate .references.json records manifest hashes, variant indices, IDs, mock state and receipt references. Pricing, sale assumptions, overhead, review provenance and account/credit denomination require their existing explicit evidence. Zero mock credits are a free TEST, not a live-serving price. This importer reads supplied manifests/bytes; it neither authenticates their authors nor polls the provider to refresh billing. Pending/final charge updates require explicit receipt recovery/reconciliation, not automatic spending or appending duplicate projects.

Dashboard generation uses the same pipeline, so new manifests carry these fields automatically for configured Magic Hour mock/future authorized live pure_prompt runs. The economics import is a researcher CLI, not a new dashboard billing system or evaluated pricing panel. No provider generation is authorized by this feature.

## Validation

97 focused checks passed after correcting an initial test fixture that accidentally supplied an all-foreground mask (the production cutout correctly refused it). Initial failed log is retained, not hidden. New cases cover final vs submission costs, missing/invalid charges, pending timeouts, provider errors, failed downloads, QA partial artifacts, stale state after prior success, transport ambiguity, mock zero, changed artifacts, plan mismatch, repeated projects, required full-variant mapping and CLI round-trip/no overwrite.

New complete default collection then ran in four whole-module processes:3,617passed,two explicit skips,zero failures/errors;3,619nodeIDs across182modules reconciled exactly once,source/tests/scripts hashes unchanged.186.72seconds supervisor total,601warnings. Skips remain missing Torch and unsupported grayscale/sRGB profile construction. This is exhaustive default-collection coverage, not single-process ordering/global-state coverage or live/GPU execution. Remote network was blocked except localhost; model downloads disabled and providerHTTP mocked. No paid call, model download, account change, default paid enablement or new photographic acceptance.

The real-photo20source/8polishedtemplate/2matchedspecialist/quality/value/currentprice+margin gaps remain open. Productization remains NO-GO.

## Reliability follow-up, 14 November 2026, 12:00 IST
New exports also include .complete.json and evidence_pair_required=true. The economics CLI requires complete-pair validation; Python from_payload needs evidence_path for these exports. Interrupted stages remain recoverable without overwrite. The previous two-file output description and full-regression result above describe the earlier snapshot. See PAIR_EXPORT_RECOVERY_2026-11-14.md for commands, scope and 102 focused checks on the new snapshot.
