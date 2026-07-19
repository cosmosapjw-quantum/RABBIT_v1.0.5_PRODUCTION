"""Clean constant check: COLD neutrino spectrum (single-signed C>0, pure heating),
compare energy-transfer moment dQ = int q^3 C dq between first-principles ground
truth and the code, at T=2,4. Ratio (code/truth) reveals: (a) T-power (should be
~const/T -> confirms one missing T), (b) the residual constant once T is removed."""
import numpy as np
from rabbit.collisions.kernels import G_F_MEV, G_L_NUE, G_R_NUE
from rabbit.collisions.deterministic_reference import (
    build_fixed_collision_quadrature, evaluate_nue_scattering_reference)

GF,GL,GR = G_F_MEV,G_L_NUE,G_R_NUE
S=0.9  # T_nu/T_gamma (colder neutrinos)
def fFD(E,T): return 1.0/(np.exp(np.minimum(E/T,500.0))+1.0)
def f_nu(E,T): return fFD(E, S*T)         # cold neutrino (1 and 3)

def E3E4(E1,E2,c12,cs):
    s=2*E1*E2*(1.0-c12)
    if s<=0: return None
    Etot=E1+E2; P=np.sqrt(max(E1*E1+E2*E2+2*E1*E2*c12,0.0))
    Ecm=np.sqrt(s); beta=P/Etot if Etot>0 else 0; gamma=Etot/Ecm; pcm=Ecm/2.0
    E3=gamma*(Ecm/2.0+beta*pcm*cs); return s,E3,Etot-E3

def C_truth(E1,T,nE2=160,nc=80,ncs=80):
    f1=f_nu(E1,T)
    E2g=np.linspace(1e-4,25*T,nE2); c12g=np.linspace(-0.999,0.999,nc); csg=np.linspace(-1,1,ncs)
    tot=0.0; dE2=E2g[1]-E2g[0]; dc=c12g[1]-c12g[0]
    for E2 in E2g:
        f2=fFD(E2,T)
        for c12 in c12g:
            s=2*E1*E2*(1.0-c12)
            if s<=1e-30: continue
            vals=np.zeros_like(csg)
            for k,cs in enumerate(csg):
                r=E3E4(E1,E2,c12,cs)
                if r is None: continue
                s_,E3,E4=r
                if E3<=0 or E4<=0: continue
                u=-(s/2.0)*(1.0+cs)
                M2=8*GF**2*(GL**2*s**2+GR**2*u**2)
                F=f_nu(E3,T)*fFD(E4,T)*(1-f1)*(1-f2)-f1*f2*(1-f_nu(E3,T))*(1-fFD(E4,T))
                vals[k]=M2*F
            tot+=(E2/(2*(2*np.pi)**2))*(1.0/(8*np.pi))*0.5*np.trapezoid(vals,csg)*dc*dE2
    return tot/(2*E1)

quad=build_fixed_collision_quadrature(N_q=24,N_nue_y2=48,N_nue_y3=24,N_pair_y2=24,N_pair_leg=16)
q=np.asarray(quad.q_nodes); qw=np.asarray(quad.q_weights)
plain=qw*np.exp(np.minimum(q,500.0))
for T in (2.0,4.0):
    f1_grid=f_nu(q*T,T)
    Ccode=np.asarray(evaluate_nue_scattering_reference(f1_grid,quadrature=quad,T_MeV=T,species="nue").C)
    Cg=np.array([C_truth(qi*T,T) for qi in q])
    dQ_code=float(np.sum(plain*q**3*Ccode))
    dQ_truth=float(np.sum(plain*q**3*Cg))
    r=dQ_code/dQ_truth
    print(f"T={T}: dQ_code={dQ_code:+.4e} dQ_truth={dQ_truth:+.4e}  code/truth={r:+.5f}  *T={r*T:+.5f}")
    # also node-wise ratio (single-signed now)
    with np.errstate(divide='ignore',invalid='ignore'):
        rn=(Ccode/Cg)*T
    print("   nodewise code/truth*T (i=3..12):", np.array2string(rn[3:13],precision=3,floatmode='fixed'))
