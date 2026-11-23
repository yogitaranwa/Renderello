"""Framework-neutral FlowEdit Euler core, n_min=0 only.

Algorithm 1: https://arxiv.org/html/2412.08629
Direct source/target velocity differences, not image conditioning or inversion.
No models, image IO, masks, RGB paste, payment or quality acceptance here.
"""
from __future__ import annotations
import math


def integrate(source, sigmas, velocity, noise_like, *, n_max, n_avg=1,
              validate_array, on_step=None):
    """Same numeric core for a NumPy oracle and a separately checked Torch run.

    velocity(array, sigma, branch, step_index) returns source/target velocity.
    noise_like(source, step_index, draw_index) returns a fresh Gaussian sample.
    validate_array must reject wrong shape/type/nonfinite arrays. It is called
    before any model call and on model outputs and states. Caller owns model
    scheduling, CFG, dtype, token handling, latent encoding and generator.
    These callbacks are trusted local code, never loaded from plan content.
    """
    if not callable(validate_array) or not callable(velocity) or not callable(noise_like):
        raise ValueError('explicit validation, velocity and noise callbacks required')
    try: schedule=list(sigmas)
    except TypeError: raise ValueError('finite decreasing sigma sequence required') from None
    if not 2<=len(schedule)<=501:raise ValueError('one to 500 integration steps required')
    if any(isinstance(s,bool) or not isinstance(s,(int,float)) or not math.isfinite(s) for s in schedule):
        raise ValueError('finite numeric sigmas required')
    if schedule[0]>1 or schedule[-1]!=0 or any(a<=b for a,b in zip(schedule,schedule[1:])):
        raise ValueError('strictly decreasing sigmas in [0,1] ending at zero required')
    steps=len(schedule)-1
    if type(n_max)is not int or not 0<=n_max<=steps:raise ValueError('bounded integer n_max required')
    if type(n_avg)is not int or not 1<=n_avg<=16:raise ValueError('bounded integer n_avg required')
    if on_step is not None and not callable(on_step):raise ValueError('step callback required')
    validate_array(source);state=source+0
    for i in range(steps-n_max,steps):
        sigma=schedule[i];delta=source*0
        for draw in range(n_avg):
            noise=noise_like(source,i,draw);validate_array(noise)
            src=(1-sigma)*source+sigma*noise
            # This order gives an exact zero displacement when state==source.
            tar=src+(state-source)
            validate_array(src);validate_array(tar)
            vs=velocity(src,sigma,'source',i);validate_array(vs)
            vt=velocity(tar,sigma,'target',i);validate_array(vt)
            delta=delta+(vt-vs)/n_avg
        state=state+(schedule[i+1]-sigma)*delta;validate_array(state)
        if on_step is not None:on_step(i,sigma,state)
    return state
