# Decisions

Why the code looks the way it does. New entries go at the bottom with a date.

## Product and pipeline

1. **Composite, never repaint, the product.** Generative whole-image edits warp labels, logos and proportions, so the deterministic routes paste verified product pixels onto a generated or procedural scene. The preservation contract covers fully opaque pixels bit-exact; semi-transparent edges and mask semantics are excluded and go to human review.
2. **Providers generate scenes only.** A provider that can touch the product breaks the contract, so the provider interface produces environments and `compose` owns the product. Whole-image Magic Hour edits exist as an explicit route flagged with the fidelity risk, never as the default.
3. **No silent provider fallback.** Callers pick the engine explicitly and the manifest records which provider produced each output. A quiet fallback would make benchmarks and cost records meaningless.
4. **GrabCut stays as the CPU cutout fallback.** It keeps the pipeline runnable and testable with zero downloads. BiRefNet (MIT) is the default model backend; RMBG-2.0 is restricted-license and stays opt-in.
5. **Twelve templates are data, not code paths.** Presets carry prompt, aspect, framing, shadow spec and knobs, so a new template never touches pipeline logic.
6. **Reference photos steer style only.** They are merged into a style descriptor for the scene and prompt and are never composited.

## Money and providers

7. **Mock by default for Magic Hour.** Without the key and the live flag, calls are served by an offline deterministic mock recording zero credits, so tests and demos can never spend.
8. **Reserve before call, reconcile after.** A failed or unreachable reserve fails closed. Afterward the reservation moves to the observed charge, up or down, because the listed price is a lowest-resolution floor, not a ceiling.
9. **Durable ledger in an isolated private GitLab project.** The on-host ledger sits on ephemeral storage and resets on redeploy; the durable copy holds cumulative committed spend behind compare-and-swap writes, with the hard cap also anchored to the live account balance so it survives deploys.
10. **The whole batch must fit before any work.** `_enforce_budget` prices templates x variants x scene attempts (plus harmonize calls on hybrid) and refuses with 402 before uploads or spend.

## Web app and security

11. **Hash-only password storage.** PBKDF2-HMAC-SHA256 at 600,000 iterations; the plaintext never exists server-side. The email compare is constant-time and the slow hash always runs so timing does not reveal whether the email matched.
12. **Stateless HMAC sessions.** Render restarts on every env change; signed tokens keep sessions alive across deploys. Logout revokes by recording the token's SHA-256 id, never the raw token.
13. **Global login rate limit instead of per-IP.** Per-client-IP attribution is unverified behind the platform proxy and forwarded headers are spoofable without a trusted-proxy basis. The honest tradeoff: the breaker bounds the total attempt rate, a per-email bucket bounds per-account guessing, and the limitation is documented rather than hidden.
14. **One job at a time on the free host.** Two concurrent in-process generations OOM-restarted the 512 MB instance on 21 November 2026. The cap fails safe to 1; values 2 to 8 are an explicit operator choice for bigger hosts.
15. **Bounded persistent worker pool.** Thread-per-job showed step-wise RSS growth under burst load; the pool stays flat and its non-daemon threads let a running job finish before process exit.
16. **24 h janitor for runs and uploads.** Free-host disk is small and ephemeral; generated results are temporary by design and downloaded copies are the durable ones.
17. **Filtered error messages.** Only known-safe prefixes reach the client; everything else is the generic "generation failed", so internals never leak through errors.
18. **CSP behind a flag.** Enforcing when `RENDERELLO_CSP_ENFORCE=1` (set on the deployment), report-only otherwise, so a settling frontend cannot be broken by accident.

## Process

19. **Human review is the acceptance gate.** Automated QA scores and development-side ratings are proxies. The seven-dimension sheet and anonymized review packs exist so a human rater can score real outputs with provenance recorded; development-side scores are never relabelled as human evidence.
20. **Failures stay visible.** Manifests list failed variants, galleries keep weak results, and review denominators include failures. Dropping them would inflate every metric.
21. **Pins track the deployed build.** `requirements.txt` equals the deployed build's resolved version set (verified against the install log), not a bit-for-bit reproducibility claim; bumps are deliberate.
