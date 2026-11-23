"""Content-bound source records. Rights metadata is not permission or clearance."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path


def identify_source(path: Path) -> dict:
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    registry = Path(__file__).with_name('source_registry.json')
    records = json.loads(registry.read_text()) if registry.is_file() else []
    match = next((r for r in records if r['sha256'] == digest), None)
    if match:
        return dict(match, match_basis='exact_file_sha256',
                    capture_origin=match.get('capture_origin', 'UNVERIFIED'),
                    physical_photo_eligible=match.get('physical_photo_eligible'),
                    rights_status='Recorded attribution only; no current legal clearance. Private test use; verify rights before republication.')
    return {'sha256': digest, 'classification': 'UNKNOWN / unclassified source',
            'match_basis': 'no_registered_content_match',
            'capture_origin': 'UNVERIFIED', 'physical_photo_eligible': None,
            'rights_status': 'No source or rights record. Do not assume synthetic, licensed, or publishable.'}
