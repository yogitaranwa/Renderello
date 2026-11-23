"""Conservative completion of saved runs, including pre-status manifests.

Only executor failure markers are failures here. Product quality warnings are
not execution failures, and an existing file alone does not prove completion.
"""
def saved_variant_failed(variant):
    if variant.get('success') is False:
        return True
    qa = variant.get('qa') or {}
    return any(str(error).startswith(('variant failed:', 'cutout failed:'))
               for error in (qa.get('critical_errors') or []))
