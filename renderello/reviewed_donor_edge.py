"""Bounded RGB transfer from explicitly reviewed same-material near-opaque donors.

Research hypothesis, not optical recovery. Alpha and all non-target RGB stay exact.
"""
import numpy as np
from scipy.ndimage import distance_transform_edt
from .candidate_diagnostics import native_rgb,native_alpha,digest
from .reviewed_model_edge import import_reviewed_model_edge


def transfer_reviewed_donor_edge(rgb,alpha,target,donor,protected,*,donor_sha256,
        max_donor_distance=24,**controls):
    native_rgb(rgb);native_alpha(alpha,rgb.shape[:2])
    if alpha.size>1_000_000:raise ValueError('1MP source budget')
    if not isinstance(target,np.ndarray)or target.dtype!=np.bool_ or target.shape!=alpha.shape or not target.any():raise ValueError('nonempty native boolean target required')
    if not isinstance(donor,np.ndarray)or donor.dtype!=np.bool_ or donor.shape!=alpha.shape or not donor.any():raise ValueError('nonempty native boolean donors required')
    if digest(donor)!=donor_sha256:raise ValueError('donor binding mismatch')
    if np.any(donor&target)or np.any(alpha[donor]<254):raise ValueError('disjoint reviewed same-material donors require alpha>=254')
    if type(max_donor_distance)is not int or not 1<=max_donor_distance<=48:raise ValueError('integer donor distance1..48')
    distances,indices=distance_transform_edt(~donor,return_indices=True)
    if np.any(distances[target]>max_donor_distance):raise ValueError('target has no nearby reviewed donor')
    estimate=rgb.copy();estimate[target]=rgb[indices[0][target],indices[1][target]]
    if 'estimate_sha256'in controls:raise ValueError('estimate is derived, not supplied')
    out,report=import_reviewed_model_edge(rgb,alpha,estimate,target,protected,estimate_sha256=digest(estimate),**controls)
    report.update(schema='reviewed-donor-edge.v1',donor_sha256=digest(donor),max_donor_distance=max_donor_distance,estimate_origin='nearest explicitly reviewed same-material near-opaque source donor',limitations=report['limitations']+['Alpha>=254 donor support is not proof of physical opacity. Donor RGB transfer can flatten real shading or erase genuine boundary colour. Nearest donor pairing is not material or lighting truth.'])
    return out,report
