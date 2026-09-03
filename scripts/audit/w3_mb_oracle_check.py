import numpy as np
from numpy.polynomial.legendre import leggauss
from numpy.polynomial.laguerre import laggauss

# Standalone verification of the W3 per-row MB closed-form oracles.
# No repo import. Reproduces the code's K_s and azimuth-averaged K_t (neutrino_self_spectral.rs
# :267-297) and integrates the massless MB loss (f=e^-y, no blocking) exactly like
# tagged_massless_mb_loss (:825-841), comparing to the claimed closed forms.

PI = np.pi
yq, wq = laggauss(64)            # int_0^inf e^-y2 g(y2) dy2 = sum wq*g(yq)  (weight e^-y2 built in)
mq, wm = leggauss(40)            # mu12 in [-1,1]
zq, wz = leggauss(40)            # z* in [-1,1]

def Ks(y1,y2,mu):
    s = 2*y1*y2*(1-mu)
    return 0.25*s*s

def Kt_azavg(y1,y2,mu,z):
    B2 = (y1-y2)**2 + 2*y1*y2*(1+mu)
    chi = (y1-y2)/np.sqrt(B2)
    one_m_chi2 = 2*y1*y2*(1+mu)/B2
    s = 2*y1*y2*(1-mu)
    bracket = (1+chi*z)**2 + 0.5*one_m_chi2*(1-z*z)
    return s*s*bracket/16.0

def mb_loss_coeff(y1, kernel, c):
    # returns loss/(G_F^2 T^5 y1 e^-y1) ; prefactor = c/(256 pi^3 y1); weight e^-(y1+y2)
    tot = 0.0
    for j,(y2,w2) in enumerate(zip(yq,wq)):
        for mu,wmu in zip(mq,wm):
            for z,wzz in zip(zq,wz):
                K = kernel(y1,y2,mu,z) if kernel is Kt_azavg else kernel(y1,y2,mu)
                # laguerre weight wq already carries e^-y2; multiply e^-y1 explicitly
                tot += w2*y2*wmu*wzz*K*np.exp(-y1)
    pref = c/(256*PI**3*y1)
    loss = pref*tot
    return loss/(y1*np.exp(-y1))

print(f"{'row':<22}{'c':>5}{'kernel':>4}{'numeric':>16}{'closed-form':>16}{'rel.err':>12}")
cases = [
    ("i  nu_a nu_a (self)", 64, Ks,       64/(8*PI**3)),
    ("iii nu_a nu_b same",  32, Ks,       32/(8*PI**3)),
    ("ii  nu_a nubar_a",   128, Kt_azavg,128/(24*PI**3)),
    ("iv  nu_a nubar_b",    32, Kt_azavg, 32/(24*PI**3)),
    ("v   nunubar->nunubar",32, Kt_azavg, 32/(24*PI**3)),
]
for name,c,ker,cf in cases:
    vals = [mb_loss_coeff(y1,ker,c) for y1 in (0.5,1.0,2.0,3.0)]
    v = np.mean(vals); spread = np.ptp(vals)/v
    print(f"{name:<22}{c:>5}{'Kt' if ker is Kt_azavg else 'Ks':>4}{v:>16.10f}{cf:>16.10f}{abs(v-cf)/cf:>12.2e}  (y-spread {spread:.1e})")
print(f"\nKt/Ks ratio at fixed c=32: {(32/(24*PI**3))/(32/(8*PI**3)):.6f}  (expect 1/3)")
print("azimuth check: int_-1^1 [(1+chi z)^2 + 0.5(1-chi^2)(1-z^2)] dz for chi in {-.9,-.3,.3,.9}:")
for chi in (-0.9,-0.3,0.3,0.9):
    I = sum(w*((1+chi*z)**2+0.5*(1-chi*chi)*(1-z*z)) for z,w in zip(zq,wz))
    print(f"   chi={chi:+.1f}  int={I:.12f}  (expect 8/3={8/3:.12f})")
