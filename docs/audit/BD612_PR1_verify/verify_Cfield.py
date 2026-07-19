import numpy as np
from rabbit.collisions.kernels import G_F_MEV, G_L_NUE, G_R_NUE
from rabbit.collisions.deterministic_reference import (
    build_fixed_collision_quadrature, evaluate_nue_scattering_reference)

GF, GL, GR = G_F_MEV, G_L_NUE, G_R_NUE
PERT = 1.3   # neutrino occupancy perturbation (nu 1 and 3 share this dist)
def fFD(E,T): return 1.0/(np.exp(np.minimum(E/T,500.0))+1.0)
def f_nu(E,T): return np.clip(fFD(E,T)*PERT,0,1)   # neutrinos 1,3 (perturbed)

def E3E4(E1,E2,c12,cs):
    s=2*E1*E2*(1.0-c12)
    if s<=0: return None
    Etot=E1+E2; P=np.sqrt(max(E1*E1+E2*E2+2*E1*E2*c12,0.0))
    Ecm=np.sqrt(s); beta=P/Etot if Etot>0 else 0; gamma=Etot/Ecm; pcm=Ecm/2.0
    E3=gamma*(Ecm/2.0+beta*pcm*cs); E4=Etot-E3
    return s,E3,E4

def C_truth(E1,T,nE2=140,nc=70,ncs=70):
    f1=f_nu(E1,T)
    E2g=np.linspace(1e-4,25*T,nE2); c12g=np.linspace(-0.999,0.999,nc); csg=np.linspace(-1,1,ncs)
    tot=0.0; dE2=E2g[1]-E2g[0]; dc=c12g[1]-c12g[0]
    for E2 in E2g:
        f2=fFD(E2,T)   # electron 2 (FD)
        for c12 in c12g:
            s=2*E1*E2*(1.0-c12)
            if s<=1e-30: continue
            vals=np.empty_like(csg)
            for k,cs in enumerate(csg):
                r=E3E4(E1,E2,c12,cs)
                if r is None: vals[k]=0; continue
                s_,E3,E4=r
                if E3<=0 or E4<=0: vals[k]=0; continue
                f3=f_nu(E3,T)      # neutrino 3 (perturbed)
                f4=fFD(E4,T)       # electron 4 (FD)
                u=-(s/2.0)*(1.0+cs)
                M2=8*GF**2*(GL**2*s**2+GR**2*u**2)
                F=f3*f4*(1-f1)*(1-f2)-f1*f2*(1-f3)*(1-f4)
                vals[k]=M2*F
            ang=0.5*np.trapezoid(vals,csg)
            tot+=(E2/(2*(2*np.pi)**2))*(1.0/(8*np.pi))*ang*dc*dE2
    return tot/(2*E1)

quad=build_fixed_collision_quadrature(N_q=24,N_nue_y2=48,N_nue_y3=24,N_pair_y2=24,N_pair_leg=16)
q=np.asarray(quad.q_nodes)
for T in (2.0,4.0):
    f1_grid=f_nu(q*T,T)
    Ccode=np.asarray(evaluate_nue_scattering_reference(f1_grid,quadrature=quad,T_MeV=T,species="nue").C)
    print(f"--- T={T} MeV ---")
    for i in (6,10,14):
        E1=q[i]*T; Cg=C_truth(E1,T)
        r=Ccode[i]/Cg if Cg!=0 else float('nan')
        print(f" E1={E1:7.3f}  C_code={Ccode[i]:+.4e}  C_truth={Cg:+.4e}  code/truth={r:+.4f}  *T={r*T:+.4f}")

print("\n=== detailed-balance sanity: PERT=1 (equilibrium) -> C_truth should ~0 ===")
import rabbit  # noop
def C_truth_eq(E1,T,nE2=140,nc=70,ncs=70):
    global PERT
    return C_truth(E1,T,nE2,nc,ncs)
PERT=1.0
for T in (2.0,):
    for E1 in (4.0, 10.0):
        print(f" T={T} E1={E1}: C_truth(eq)={C_truth(E1,T):+.3e} (want ~0)")

print("\n=== finer low-q ratio stability (PERT=1.3, hi-res) ===")
PERT=1.3
quad=build_fixed_collision_quadrature(N_q=24,N_nue_y2=48,N_nue_y3=24,N_pair_y2=24,N_pair_leg=16)
q=np.asarray(quad.q_nodes)
for T in (2.0,):
    f1_grid=f_nu(q*T,T)
    Ccode=np.asarray(evaluate_nue_scattering_reference(f1_grid,quadrature=quad,T_MeV=T,species="nue").C)
    for i in (1,2,3,4,5,6,8):
        E1=q[i]*T; Cg=C_truth(E1,T,nE2=220,nc=110,ncs=110)
        r=Ccode[i]/Cg if Cg!=0 else float('nan')
        print(f" i={i:2d} q={q[i]:6.3f} E1={E1:7.3f}  code/truth*T={r*T:+.4f}")
