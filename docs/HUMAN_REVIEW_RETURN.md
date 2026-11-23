# Returned human review sheet workflow

14 November 2026, 09:15 IST. Local implementation only. No independent rater or successful specialist comparison is claimed. (Terminology note: this workflow prepares anonymized packs for human review; the code modules keep their original `blind_review` names.)

Existing review packs previously sealed the REVIEW files, including blank scores.csv, but not the PRIVATE method mapping. Editing the sealed blank sheet made verification fail and there was no returned-score ingestion path into typed acceptance records. The new workflow preserves the original pack and imports a separate returned copy.

1. Build a new pack with `blind_review.build_pack`. New schema-2 packs hash the private mapping as well as all review assets. Original schema-1 packs remain readable through verify_pack but are not silently upgraded or accepted by the new importer.
2. Only share REVIEW, never PRIVATE or a returned private report. This implementation does not send or share anything. The private seed/key and method names must stay away from the human rater. A randomized token sheet is not proof that a reviewer was independent or unaware of model identities.
3. A rater fills a separate copy of scores.csv. Keep columns and generation status unchanged. Each token must occur once, including failures. Seven scores use integer 1-5. `no_critical` is true or false; `critical_errors` is a JSON string array, including explicit [] when none. State the rater identifier. Missing fields remain incomplete, not zero or inferred acceptance. Failed generations cannot receive image scores.
4. Import locally:

```
python -m renderello.blind_review_return --pack pack-directory \
  --returned-csv separate-return.csv --provenance claimed_independent \
  --review-reference review-session-reference --output new-private-report.json
```

Caller provenance must come from the actual trusted context, not the rater name typed in the CSV. `claimed_independent` deliberately records a claim, not identity verification. Owner and development-side judgments must remain labelled as such. The output contains the private method key and is not a rater-facing artifact. Exclusive creation prevents replacing an earlier report; output/returned sheet must stay outside the sealed pack.

5. For one method's preregistered plan, call `bind_ratings(report, plan, attempts, token_output_ids)`. The explicit token-to-output mapping must agree with the native output SHA256, source SHA256, method and recorded successful attempt. Existing first-try constraints apply. This joins to `brief_acceptance.evaluate` or `brief_economics.build_report`; it does not guess the template/alternative from filenames or copy one score to several slots. Missing reviews/failures remain in the original plan denominator.

The pack additionally rejects identical decoded output pixels saved with different PNG encodings, rather than mistaking metadata changes for a distinct candidate. It does not equate byte/pixel difference with visual variety or polished template quality. Existing exact-dimension registration requirement remains; never resize a result and pretend its native fine print was reviewed.

152 related checks passed in 3.62 seconds across seven modules, with eleven existing Pillow deprecation warnings. Tests use synthetic images and synthetic ratings. This is neither a full-project regression pass nor a photographic-quality result. No external communications, new review claims, paid calls or service signup occurred.
