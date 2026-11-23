"""Decide which saved candidates need exact-fidelity review, never acceptance.

Generative product edits need review even if every heuristic is quiet. Flags
are conservative warnings, not semantic proof. All outputs remain downloadable.
"""
GENERATIVE_PRODUCT_ROUTES = frozenset(('pure_prompt', 'hybrid'))


def review_requirement(route, flags=(), *, output_exists=True, completed=True):
    if not completed:
        return dict(required=False, status='FAILED_PARTIAL_ARTIFACT' if output_exists else 'FAILED_NO_OUTPUT', reasons=[],
                    product_text_verified=False, photo_accepted=False)
    if not output_exists:
        return dict(required=False, status='NOT_GENERATED', reasons=[],
                    product_text_verified=False, photo_accepted=False)
    reasons = []
    if route in GENERATIVE_PRODUCT_ROUTES:
        reasons.append('GENERATIVE_PRODUCT_EDIT_ALWAYS_REQUIRES_FIDELITY_REVIEW')
    if flags:
        reasons.append('AUTOMATED_WARNING_REQUIRES_REVIEW')
    # Quiet deterministic composites also have unverified alpha/material/contact.
    # Distinguish that general quality review from the product-redraw queue.
    return dict(required=bool(reasons),
                status='EXACT_FIDELITY_REVIEW_REQUIRED' if reasons else 'NO_TRIPWIRE_NOT_A_QUALITY_PASS',
                reasons=reasons, product_text_verified=False, photo_accepted=False)


def review_caption(requirement):
    if requirement.get('status') in ('FAILED_PARTIAL_ARTIFACT', 'FAILED_NO_OUTPUT'):
        return 'FAILED: any saved image is a debug artifact, not a completed output or quality-rated candidate.'
    if requirement.get('status') == 'NOT_GENERATED':
        return 'NOT GENERATED: no image to review.'
    if 'GENERATIVE_PRODUCT_EDIT_ALWAYS_REQUIRES_FIDELITY_REVIEW' in requirement.get('reasons', []):
        return ('FIDELITY REVIEW REQUIRED: this route generates product pixels. '
                'Zero automated warnings does not verify label text, shape or logos. '
                'Compare the original and full-size output before use.')
    if requirement.get('required'):
        return 'REVIEW REQUIRED: automated warnings need source/output inspection, not automatic acceptance.'
    return 'No automated tripwire. Mask, material, label readability and publication quality remain unverified.'
