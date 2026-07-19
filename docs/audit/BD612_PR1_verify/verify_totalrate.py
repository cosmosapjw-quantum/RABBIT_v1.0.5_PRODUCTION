"""Ground-truth benchmark: thermal-averaged total nu_e + e- -> nu_e + e- scattering
rate from first principles, checked against the repo's total_rate = (7 pi/12) G_F^2 T^5 (G_L^2+G_R^2).

Massless e. |M|^2 summed = 32 G_F^2 [ G_L^2 (p1.p2)(p3.p4) + G_R^2 (p1.p4)(p2.p3) ]
for nu(1) e-(2) -> nu(3) e-(4). Reduce final state via CM 2-body phase space:
  Gamma_loss(E1) = 1/(2 E1) * int d^3p2/((2pi)^3 2E2) f_e(E2) *
                   [ int dPi3 dPi4 (2pi)^4 delta^4 |M|^2 ] * (Pauli-blocking ~1 for rate scale)
  final-state bracket = (1/(8 pi)) * (1/2) int_{-1}^{1} dcos* |M|^2(s,t)
We compute Gamma_loss(E1) WITHOUT blocking factors (pure rate normalization) and
thermally average <Gamma> = int f_nu(E1) Gamma(E1) E1^2 dE1 / int f_nu E1^2 dE1 ...
Actually simplest: verify the ENERGY-independent coefficient by computing
Gamma(E1)/E1 -> constant * G_F^2 T^4 form, and the number-rate scaling ~ G_F^2 T^5.
Here we check the T-scaling and the (7 pi/12) coefficient of the total rate
n_e <sigma v> style integral.
"""
import numpy as np
from scipy import integrate

GF = 1.0            # set G_F=1, restore later; couplings absorbed
GL2 = 1.0
GR2 = 0.0           # take pure G_L (left) channel first to isolate the coefficient

def fFD(E, T):
    return 1.0/(np.exp(E/T)+1.0)

# For nu(1)+e(2)->nu(3)+e(4) massless, s=2 p1.p2 = 2 E1 E2 (1-cos12).
# |M|^2 (G_R=0): 32 GF^2 GL^2 (p1.p2)(p3.p4). In CM, (p1.p2)=s/2, (p3.p4)=s/2 => |M|^2=32 GF^2 GL^2 s^2/4 = 8 GF^2 GL^2 s^2 (t-independent for GL channel!)
# So int dcos* |M|^2 = 2 * 8 GF^2 GL^2 s^2 = 16 GF^2 GL^2 s^2 ; (1/2)int = 8 GF^2 GL^2 s^2
# final bracket = (1/(8pi)) * 8 GF^2 GL^2 s^2 = GF^2 GL^2 s^2 / pi
# Gamma(E1) = 1/(2E1) int d^3p2/((2pi)^3 2E2) f2 * GF^2 GL^2 s^2/pi
#           = 1/(2E1) * (1/(2pi)^2) * (1/2) int p2 dp2 dcos12 f2 * GF^2 GL^2 s^2/pi   [d^3p2=2pi p2^2 dcos dp2, /2E2=/2p2]
# s=2E1 p2 (1-cos12) (massless, E2=p2). Let u=1-cos12 in [0,2].
def gamma_of_E1(E1, T, n=400):
    p2 = np.linspace(1e-6, 40*T, n)
    # integrate over cos12 analytically: int_{-1}^{1} s^2 dcos = (2E1 p2)^2 int u^2?
    # s^2 = (2 E1 p2)^2 (1-cos)^2 ; int_{-1}^{1}(1-cos)^2 dcos, let x=cos: int_{-1}^1 (1-x)^2 dx = [ -(1-x)^3/3 ]_{-1}^{1}=(0 - (-8/3))=8/3
    s2_cosint = (2*E1*p2)**2 * (8.0/3.0)
    integrand = p2 * fFD(p2, T) * s2_cosint
    I = integrate.simpson(integrand, p2)
    pref = 1.0/(2*E1) * (1.0/(2*np.pi)**2) * 0.5 * (GF**2*GL2/np.pi)
    return pref * I

# number-loss rate averaged over the incident neutrino spectrum, per neutrino:
def thermal_rate(T, n=200):
    E1 = np.linspace(1e-6, 30*T, n)
    w = fFD(E1, T)*E1**2
    g = np.array([gamma_of_E1(e, T) for e in E1])
    return integrate.simpson(w*g, E1)/integrate.simpson(w, E1)

for T in (1.0, 2.0, 4.0):
    G = thermal_rate(T)
    print(f"T={T}: <Gamma>={G:.6e}  <Gamma>/T^5={G/T**5:.6f}  (7pi/12)={7*np.pi/12:.6f}")
