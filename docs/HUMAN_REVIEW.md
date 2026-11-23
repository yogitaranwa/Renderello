# Seven-dimension review sheet

This is preparation, not a completed benchmark. No scores are pre-filled.

Export existing outputs (original run paths must be accessible):

    python -m renderello.cli review-export --runs runs/run1 runs/run2 --out review/scores.csv

The exporter refuses to overwrite a sheet, records each output SHA-256, and
skips missing files. Generation failures stay in run manifests; skipping a
missing file here does not remove it from the benchmark denominator.

Fill every dimension with an integer1-5, with5 best:

- product_label_fidelity: original shape, color, logos and label typography
- photorealism: believable photographic result
- shadow_contact_realism: grounded contact, direction and shadow consistency
- material_edges: glass, reflections, fur and boundary quality
- composition: framing, scale and placement
- template_distinctness: visible match to the intended shoot style
- publish_ready_usefulness: commercially usable as delivered

Set no_critical to exactly true or false. When false, list the critical error
in critical_errors, such as changed logo, invented text, wrong product or
deformed packaging. When true, critical_errors must be blank. Optional notes
can record noncritical weaknesses.

Set rater to owner, independent_human or agent_eye. A CSV label is only a
self-report. It does not prove identity, that a human performed the review,
or permission to act. Verify the rater through the original trusted channel
before counting it as human evidence. Imported records explicitly carry
provenance_verified=false. Agent-eye review must never become a human score.

Check the completed sheet without generation, network calls or acceptance claims:

    python -m renderello.cli review-check --sheet review/scores.csv

The checker rejects blank, fractional, out-of-range and conflicting scores,
duplicate outputs, changed/missing image bytes and absent rater labels. The
load_scores function returns the seven dimensions and the publishability value
in the benchmark rating shape. It does not automatically merge ratings into an
acceptance report. Caller must verify provenance and full output coverage first.

Full acceptance also needs 20+ distinct real sources, 8+ templates and the matched
specialist comparisons. A fully scored small sample does not satisfy those
requirements. No anonymized-comparison or specialist results are created here.

