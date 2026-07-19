"""
rabbit.solver.rodas5p — Rosenbrock-Wanner ODE solver with Rodas5P.

Frozen NumPy numerical-method reference; BDF remains the temporary
number-of-record while Rust is the active implementation target.
Two conventions are supported:
  - "hairer": W = I - γhJ, k ~ O(h), y_new = y + Σ b k  (ROS34PW2)
  - "sciml":  W = I/(γh) - J, k ~ O(h), y_new = y + Σ b k, C/h coupling  (Rodas5P)

Methods:
  - ROS34PW2 (order 3, 4-stage, L-stable) — validated backup
  - RODAS5P  (order 5, 8-stage, A-stable, stiffly accurate) — frozen reference

References:
    Steinebach (2023), BIT Numer Math 63:27 — Rodas5P coefficients
    Rang & Angermann (2005), BIT 45, 761 — ROS34PW2
    Hairer & Wanner (1996), Solving ODEs II — general Rosenbrock theory
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Callable, Optional, Tuple
import numpy as np
from scipy.linalg import lu_factor, lu_solve


@dataclass
class RosenbrockTableau:
    name: str; order: int; stages: int; gamma: float
    A: np.ndarray; G: np.ndarray  # G = off-diagonal γ_{ij} (Hairer) or C (SciML)
    c: np.ndarray; d: np.ndarray  # stage times and partial-time coefficients
    b: np.ndarray; bhat: np.ndarray
    convention: str = "hairer"  # "hairer" or "sciml"


# ═══════════════════════════════════════════════════════════════
# §1. RODAS5P (Steinebach 2023) — SciML convention
# ═══════════════════════════════════════════════════════════════

def _make_rodas5p():
    g = 0.21193756319429014
    A = np.zeros((8,8))
    A[1,0]=3.0
    A[2,0]=2.849394379747939;A[2,1]=0.45842242204463923
    A[3,0]=-6.954028509809101;A[3,1]=2.489845061869568;A[3,2]=-10.358996098473584
    A[4,0]=2.8029986275628964;A[4,1]=0.5072464736228206;A[4,2]=-0.3988312541770524;A[4,3]=-0.04721187230404641
    A[5,0]=-7.502846399306121;A[5,1]=2.561846144803919;A[5,2]=-11.627539656261098;A[5,3]=-0.18268767659942256;A[5,4]=0.030198172008377946
    for j in range(5):A[6,j]=A[5,j]
    A[6,5]=1.0
    for j in range(5):A[7,j]=A[5,j]
    A[7,5]=1.0;A[7,6]=1.0

    C = np.zeros((8,8))
    C[1,0]=-14.155112264123755
    C[2,0]=-17.97296035885952;C[2,1]=-2.859693295451294
    C[3,0]=147.12150275711716;C[3,1]=-1.41221402718213;C[3,2]=71.68940251302358
    C[4,0]=165.43517024871676;C[4,1]=-0.4592823456491126;C[4,2]=42.90938336958603;C[4,3]=-5.961986721573306
    C[5,0]=24.854864614690072;C[5,1]=-3.0009227002832186;C[5,2]=47.4931110020768;C[5,3]=5.5814197821558125;C[5,4]=-0.6610691825249471
    C[6,0]=30.91273214028599;C[6,1]=-3.1208243349937974;C[6,2]=77.79954646070892;C[6,3]=34.28646028294783;C[6,4]=-19.097331116725623;C[6,5]=-28.087943162872662
    C[7,0]=37.80277123390563;C[7,1]=-3.2571969029072276;C[7,2]=112.26918849496327;C[7,3]=66.9347231244047;C[7,4]=-40.06618937091002;C[7,5]=-54.66780262877968;C[7,6]=-9.48861652309627

    b = np.array([-7.502846399306121,2.561846144803919,-11.627539656261098,
                  -0.18268767659942256,0.030198172008377946,1.0,1.0,1.0])
    bt = np.array([0,0,0,0,0,0,0,1.0])
    c = np.array([0.0, 0.6358126895828704, 0.4095798393397535,
                  0.9769306725060716, 0.4288403609558664, 1.0, 1.0, 1.0])
    d = np.array([0.21193756319429014, -0.42387512638858027,
                  -0.3384627126235924, 1.8046452872882734,
                  2.325825639765069, 0.0, 0.0, 0.0])
    return RosenbrockTableau("Rodas5P",5,8,g,A,C,c,d,b,bt,"sciml")


# ═══════════════════════════════════════════════════════════════
# §2. ROS34PW2 (Rang & Angermann 2005) — Hairer convention
# ═══════════════════════════════════════════════════════════════

def _make_ros34pw2():
    g = 4.3586652150845900e-01
    A = np.zeros((4,4))
    A[1,0]=8.7173304301691801e-01
    A[2,0]=8.4457060015369423e-01;A[2,1]=-1.1299064236397981e-01
    A[3,0]=0.0;A[3,1]=0.0;A[3,2]=1.0
    G = np.zeros((4,4))
    G[1,0]=-8.7173304301691801e-01
    G[2,0]=-9.0338057013044082e-01;G[2,1]=5.4180672388095326e-02
    G[3,0]=2.4212380706095346e-01;G[3,1]=-1.2232505839045147e+00;G[3,2]=5.4526025533510214e-01
    b = np.array([2.4212380706095346e-01,-1.2232505839045147e+00,1.5452602553351023e+00,4.3586652150845900e-01])
    bh = np.array([3.7810903145819369e-01,-9.6042292212423178e-02,5.0000000000000000e-01,2.1793326075422950e-01])
    c = np.sum(A, axis=1)
    d = g + np.sum(G, axis=1)
    return RosenbrockTableau("ROS34PW2",3,4,g,A,G,c,d,b,bh,"hairer")


RODAS5P = _make_rodas5p()
ROS34PW2 = _make_ros34pw2()


# ═══════════════════════════════════════════════════════════════
# §3. Configuration
# ═══════════════════════════════════════════════════════════════

@dataclass
class Rodas5PConfig:
    rtol: float = 1e-8; atol: float = 1e-10
    max_steps: int = 10000
    h_init: Optional[float] = None
    h_min: float = 1e-14; h_max: float = 10.0
    f_safety: float = 0.9; f_min: float = 0.2; f_max: float = 6.0
    beta: float = 0.04; max_step_N: float = 0.5
    tableau: RosenbrockTableau = field(default_factory=lambda: RODAS5P)
    #: How many consecutive accepted steps may reuse a single Jacobian before a
    #: forced recompute. 1 (default) = recompute every step = exact Rosenbrock
    #: (the FD Jacobian costs N+1 RHS evals/step and dominates at large N, so
    #: reuse>1 is the main cost lever). Rosenbrock formally assumes an exact
    #: per-step Jacobian; reuse>1 is an approximation and must be endpoint-parity
    #: gated before use in any preset (BD615). A rejected step always forces a
    #: recompute regardless of this cap.
    jac_reuse_max_steps: int = 1

REFERENCE = Rodas5PConfig(rtol=1e-10,atol=1e-12,max_step_N=0.1,tableau=RODAS5P)
REPEATED_RUN = Rodas5PConfig(rtol=1e-8,atol=1e-10,max_step_N=0.5,tableau=RODAS5P)
# Historical compatibility alias only; it does not confer publication authority.
PRODUCTION = REPEATED_RUN
FAST = Rodas5PConfig(rtol=1e-6,atol=1e-8,max_step_N=1.0,tableau=RODAS5P)
FALLBACK = Rodas5PConfig(rtol=1e-8,atol=1e-10,max_step_N=0.5,tableau=ROS34PW2)


# ═══════════════════════════════════════════════════════════════
# §4. Utilities
# ═══════════════════════════════════════════════════════════════

def _err_norm(e, y, rtol, atol):
    sc = atol + rtol*np.abs(y)
    return np.sqrt(np.mean((e/sc)**2))

def _jac_fd(f, t, y, eps=1e-8):
    N=len(y);f0=f(t,y);J=np.zeros((N,N))
    for j in range(N):
        dy=np.zeros(N);dy[j]=max(abs(y[j])*eps,eps)
        J[:,j]=(f(t,y+dy)-f0)/dy[j]
    return J


def _partial_dfdt(f, t, y, h, dfdt_fn=None):
    """Return the step-start partial derivative ``∂f/∂t`` at fixed ``y``.

    The fallback scale follows the frozen PUB-02 contract.  This helper is
    called exactly once by ``_step`` and its result is shared by all stages.
    """
    if dfdt_fn is not None:
        return np.asarray(dfdt_fn(t, y))

    dtype = np.result_type(np.asarray(y).dtype, np.asarray(t).dtype, np.asarray(h).dtype)
    if not np.issubdtype(dtype, np.inexact):
        dtype = np.dtype(float)
    delta_t = np.cbrt(np.finfo(dtype).eps) * max(1.0, abs(t), abs(h))
    return (np.asarray(f(t + delta_t, y)) - np.asarray(f(t - delta_t, y))) / (
        2.0 * delta_t
    )


# ═══════════════════════════════════════════════════════════════
# §5. Step functions (convention-aware)
# ═══════════════════════════════════════════════════════════════

def _step_hairer(f, t, y, h, J, tab, cfg, dfdt):
    """Hairer convention: W=I-γhJ, rhs=hf+hJΣGk+h²d∂tf.

    The stage matrix W is constant across all s stages, so it is LU-factored
    once and each stage is a cheap triangular back-substitution (BD615): this
    replaces s O(N^3) refactorizations with one factorization + s O(N^2) solves.
    """
    N=len(y);s=tab.stages;g=tab.gamma
    W = np.eye(N) - g*h*J
    try: lu = lu_factor(W)
    except (np.linalg.LinAlgError, ValueError): return y,np.inf,False
    k=[None]*s
    for i in range(s):
        y_st=y.copy();Jg=np.zeros(N)
        for j in range(i):
            if k[j] is not None:
                y_st+=tab.A[i,j]*k[j];Jg+=tab.G[i,j]*k[j]
        rhs=h*f(t+tab.c[i]*h,y_st)+h*(J@Jg)+h*h*tab.d[i]*dfdt
        if not np.all(np.isfinite(rhs)):
            return y,np.inf,False
        try: k[i]=lu_solve(lu,rhs)
        except (np.linalg.LinAlgError, ValueError): return y,np.inf,False
    yn=y.copy()
    for i in range(s): yn+=tab.b[i]*k[i]
    if not np.all(np.isfinite(yn)): return y,np.inf,False
    ev=np.zeros(N)
    for i in range(s): ev+=(tab.b[i]-tab.bhat[i])*k[i]
    return yn, _err_norm(ev,yn,cfg.rtol,cfg.atol), True


def _step_sciml(f, t, y, h, J, tab, cfg, dfdt):
    """SciML convention: W=I/(γh)-J, rhs=f+ΣCk/h+hd∂tf, y+=Σbk.
    Error = Σ btilde_i k_i  (btilde IS the error weights in SciML)."""
    N=len(y);s=tab.stages;g=tab.gamma
    W = np.eye(N)/(g*h) - J
    try: lu = lu_factor(W)   # W constant across stages → factor once (BD615)
    except (np.linalg.LinAlgError, ValueError): return y,np.inf,False
    k=[None]*s
    for i in range(s):
        y_st=y.copy();c_sum=np.zeros(N)
        for j in range(i):
            if k[j] is not None:
                y_st+=tab.A[i,j]*k[j]     # NO h on A
                c_sum+=tab.G[i,j]*k[j]/h  # C/h
        rhs=f(t+tab.c[i]*h,y_st)+c_sum+h*tab.d[i]*dfdt
        if not np.all(np.isfinite(rhs)):
            return y,np.inf,False
        try: k[i]=lu_solve(lu,rhs)
        except (np.linalg.LinAlgError, ValueError): return y,np.inf,False
    yn=y.copy()
    for i in range(s): yn+=tab.b[i]*k[i]  # NO h on b
    if not np.all(np.isfinite(yn)): return y,np.inf,False
    # SciML: btilde = b - bhat_embedded, so error = Σ btilde*k directly
    ev=np.zeros(N)
    for i in range(s): ev+=tab.bhat[i]*k[i]
    return yn, _err_norm(ev,yn,cfg.rtol,cfg.atol), True


def _step(f, t, y, h, J, tab, cfg, dfdt_fn=None):
    dfdt = _partial_dfdt(f, t, y, h, dfdt_fn)
    if tab.convention == "sciml":
        return _step_sciml(f, t, y, h, J, tab, cfg, dfdt)
    return _step_hairer(f, t, y, h, J, tab, cfg, dfdt)


# ═══════════════════════════════════════════════════════════════
# §6. Gustafsson + Main Loop
# ═══════════════════════════════════════════════════════════════

def _new_h(h, err, ep, cfg, order):
    if err<1e-30: return min(h*cfg.f_max,cfg.h_max,cfg.max_step_N)
    fac=cfg.f_safety*(1/err)**(1/order)
    if ep>0: fac*=ep**cfg.beta
    fac=max(cfg.f_min,min(cfg.f_max,fac))
    return max(cfg.h_min,min(h*fac,cfg.h_max,cfg.max_step_N))


@dataclass
class Rodas5PResult:
    t:np.ndarray;y:np.ndarray;success:bool;n_steps:int
    n_rejected:int;n_jac:int;message:str;h_final:float;method:str=""
    n_attempts:int=0;failure_reason:Optional[str]=None


def _event_fires(ev_prev, ev_new, direction):
    """Zero-crossing test matching scipy's ``events`` direction convention.

    direction == 0 : either direction (any sign change);
    direction < 0  : positive→negative crossing (scipy includes ev_new == 0);
    direction > 0  : negative→positive crossing.
    """
    if direction == 0:
        return ev_prev * ev_new < 0.0
    if direction < 0:
        return ev_prev > 0.0 and ev_new <= 0.0
    return ev_prev < 0.0 and ev_new >= 0.0


def _refine_event(f, t, y, ht, sgn, J, tab, cfg, events, ev_prev, dfdt_fn=None):
    """Localize a single zero crossing inside an accepted step [t, t+ht*sgn].

    A plain linear interpolation of the crossing (ev_prev / (ev_prev-ev_new))
    is only first-order accurate and, on a stiff BBN handoff/end T-crossing,
    leaves an O(1e-4) offset in N that propagates into the endpoint abundances.
    This bisects the sub-step size on the event sign, taking a fresh Rosenbrock
    sub-step from (t, y) with the already-computed Jacobian J at each probe
    (~50 O(N^2) back-substitutions per event; events fire ~twice per solve, so
    the cost is negligible). Returns the refined (t_cross, y_cross).
    """
    if ev_prev == 0.0:
        return t, y.copy()
    lo, hi = 0.0, ht
    t_hi, y_hi = t + ht * sgn, None
    prev_pos = ev_prev > 0.0
    for _ in range(60):
        if (hi - lo) <= 1e-15 * max(1.0, abs(t)):
            break
        mid = 0.5 * (lo + hi)
        ym, _, ok = _step(f, t, y, mid * sgn, J, tab, cfg, dfdt_fn)
        if not ok:
            hi = mid
            continue
        ev_mid = events(t + mid * sgn, ym)
        if (ev_mid > 0.0) == prev_pos:   # still on the start side → crossing is past mid
            lo = mid
        else:                             # crossed → tighten the upper bracket
            hi, t_hi, y_hi = mid, t + mid * sgn, ym
    if y_hi is None:                      # crossing never bracketed below ht → use step endpoint
        ym, _, _ = _step(f, t, y, ht * sgn, J, tab, cfg, dfdt_fn)
        return t + ht * sgn, ym
    return t_hi, y_hi


def solve(f,t_span,y0,config=None,events=None,jac_fn=None,event_direction=0,dfdt_fn=None):
    """Integrate ``dy/dt = f(t, y)`` from ``t_span[0]`` to ``t_span[1]``.

    ``events`` is a single scalar callable ``events(t, y) -> float``; a zero
    crossing (filtered by ``event_direction``, scipy convention) terminates the
    solve and appends the linearly-interpolated crossing point. ``event_direction``
    defaults to 0 (any direction), preserving the historical any-crossing behavior.
    """
    if config is None: config=REPEATED_RUN
    tab=config.tableau; t0,tend=t_span
    y=np.array(y0,float);t=t0
    if config.h_init:
        h=config.h_init
    else:
        f0=f(t,y);sc=config.atol+config.rtol*np.abs(y)
        d0=np.sqrt(np.mean((y/sc)**2));d1=np.sqrt(np.mean((f0/sc)**2))
        h=0.01*d0/max(d1,1e-30) if d0>1e-10 else 1e-6
        h=min(h,config.max_step_N,abs(tend-t0)*0.1)
    sgn=1.0 if tend>t0 else -1.0; h=abs(h)
    th=[t];yh=[y.copy()];ns=0;nr=0;nj=0;ep=1.0;ok=True;msg="OK";failure_reason=None
    ev_prev=events(t,y) if events else None
    J=None;steps_since_jac=0;reuse=max(1,int(config.jac_reuse_max_steps))

    while sgn*(t-tend)<-1e-14 and ns+nr<config.max_steps:
        ht=min(h,abs(tend-t))
        if J is None or steps_since_jac>=reuse:
            J=jac_fn(t,y) if jac_fn else _jac_fd(f,t,y);nj+=1;steps_since_jac=0
        yn,err,step_ok=_step(f,t,y,ht*sgn,J,tab,config,dfdt_fn)
        if not step_ok or err>1.0:
            nr+=1;J=None
            if ht<=config.h_min or h<=config.h_min:
                ok=False;failure_reason="h_min";msg=f"h_min at t={t}";break
            h=_new_h(ht,max(err,2),ep,config,tab.order)  # reject → rescale attempted step
            continue
        tn=t+ht*sgn;ns+=1;steps_since_jac+=1
        if events:
            ev_new=events(tn,yn)
            if _event_fires(ev_prev,ev_new,event_direction):
                t_cross,y_cross=_refine_event(
                    f,t,y,ht,sgn,J,tab,config,events,ev_prev,dfdt_fn
                )
                th.append(t_cross);yh.append(y_cross);msg="Event";break
            ev_prev=ev_new
        th.append(tn);yh.append(yn.copy())
        h=_new_h(h,err,ep,config,tab.order);ep=err;t=tn;y=yn
    if sgn*(t-tend)<-1e-14 and msg!="Event" and failure_reason is None:
        ok=False;failure_reason="max_steps";msg=f"Max steps at t={t}"
    return Rodas5PResult(
        np.array(th),np.array(yh).T,ok,ns,nr,nj,msg,h,tab.name,
        ns+nr,failure_reason,
    )
