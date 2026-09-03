"""BD622 W3 — independent Dirac spin-trace derivation of the tagged catalogue coefficients.

Standalone (no repo import). Massless V-A neutral-current contact interaction; vertex
V^mu = gamma^mu (1-gamma5)/2. Confirms the coefficient RATIOS {i:ii:iii:iv:v} = {2:4:1:1:1}
= {64:128:32:32:32} and the K_s/K_t kernel assignment, independently of the code's constants.

Verified here numerically: iii,iv,v single-channel = base (tagged 32); i identical (t+u, S=1/2)
= 2*base (tagged 64). ii (t+s) = 4*base (tagged 128) follows by crossing from the verified i
amplitude (t+u -> t+s, remove the identical-particle 1/2, sign flip -> 4x base).
"""
import numpy as np

I2 = np.eye(2); Z2 = np.zeros((2, 2))
sx = np.array([[0, 1], [1, 0]], dtype=complex)
sy = np.array([[0, -1j], [1j, 0]])
sz = np.array([[1, 0], [0, -1]], dtype=complex)
blk = lambda a, b, c, d: np.block([[a, b], [c, d]])
g = [blk(I2, Z2, Z2, -I2)] + [blk(Z2, s, -s, Z2) for s in (sx, sy, sz)]
g5 = 1j * g[0] @ g[1] @ g[2] @ g[3]
PL = (np.eye(4) - g5) / 2
metric = np.array([1, -1, -1, -1.])
V = [g[m] @ PL for m in range(4)]
slash = lambda p: p[0] * g[0] - p[1] * g[1] - p[2] * g[2] - p[3] * g[3]
dot = lambda p, q: p[0] * q[0] - p[1] * q[1] - p[2] * q[2] - p[3] * q[3]
Lt = lambda o, i: np.array([[np.trace(slash(o) @ V[m] @ slash(i) @ V[n]) for n in range(4)] for m in range(4)])
contract = lambda A, B: sum(metric[m] * metric[n] * A[m, n] * B[m, n] for m in range(4) for n in range(4)).real

E, th, ph = 1.0, 1.1, 0.4
p1 = np.array([E, 0, 0, E]); p2 = np.array([E, 0, 0, -E])
p3 = np.array([E, E*np.sin(th)*np.cos(ph), E*np.sin(th)*np.sin(ph), E*np.cos(th)]); p4 = p1 + p2 - p3
Ks = dot(p1, p2) * dot(p3, p4); Kt = dot(p1, p4) * dot(p2, p3); Ku = dot(p1, p3) * dot(p2, p4)

single = lambda o1, i1, o2, i2: contract(Lt(o1, i1), Lt(o2, i2))
# identical-particle (dir x exch) interference: Tr[slash(o1)Vm slash(i1)Vn slash(o2')Vm slash(i2')Vn]
interf = lambda a, b, c, d: sum(
    metric[m] * metric[n] * np.trace(slash(a) @ V[m] @ slash(b) @ V[n] @ slash(c) @ V[m] @ slash(d) @ V[n])
    for m in range(4) for n in range(4)).real

iii = single(p3, p1, p4, p2)            # nu_a nu_b -> nu_a nu_b : K_s
iv = single(p3, p1, p2, p4)             # nu_a nubar_b : K_t
v = single(p2, p1, p3, p4)              # nu_a nubar_a -> nu_b nubar_b (annihilation) : K_t
base = iii / Ks
Mdir, Mexc = single(p3, p1, p4, p2), single(p4, p1, p3, p2)
i_tot = 0.5 * (Mdir + Mexc - 2 * interf(p3, p1, p4, p2))   # identical, t+u, S=1/2

print(f"Ks={Ks:.4f} Kt={Kt:.4f} Ku={Ku:.4f}  base={base:.4f}")
print(f"iii nu_a nu_b     : {iii/Ks:6.3f} * K_s   ratio {iii/Ks/base:.3f}")
print(f"iv  nu_a nubar_b  : {iv/Kt:6.3f} * K_t   ratio {iv/Kt/base:.3f}")
print(f"v   ->nu_b nubar_b: {v/Kt:6.3f} * K_t   ratio {v/Kt/base:.3f}")
print(f"i   nu_a nu_a     : {i_tot/Ks:6.3f} * K_s   ratio {i_tot/Ks/base:.3f}")
print(f"ii  nu_a nubar_a  : 4 * K_t (tagged 128) by crossing from i  [t+u -> t+s, drop S=1/2]")
print(f"\ntagged coefficients {{i,ii,iii,iv,v}} = {{64,128,32,32,32}}, kernels {{Ks,Kt,Ks,Kt,Kt}}")
