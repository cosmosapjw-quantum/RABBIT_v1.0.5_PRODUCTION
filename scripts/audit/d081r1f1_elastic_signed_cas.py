"""Exact physical-path references for elastic scalar thermal tangents.

T, m, q, E in MeV; d_ab = -c^2 g(p_a,p_b), metric (-,+,+,+).
T>0, m=3, p1=1, fixed outgoing COM angle pi/2 and outer weight 1 MeV^3.
Manufactured scalar validation only. No production configuration is changed.
Run with SymPy and mpmath; stdout is a reproducible JSON receipt.
"""

import json
import platform
import sys

import mpmath as mp
import sympy as sp

T = sp.symbols("T", positive=True)
L, R = sp.symbols("left right", real=True)
order = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
Q = sp.Rational


def dot(a, b):
    return a[0] * b[0] - sum(a[i] * b[i] for i in range(1, 4))


def path(t, sigma, sqrt):
    e2 = sqrt(4 * t * t + 9)
    et, pz = 1 + e2, 1 + sigma * 2 * t
    s = et * et - pz * pz
    delta = s - 9
    k1, k2 = [1, 0, 0, 1], [e2, 0, 0, sigma * 2 * t]
    k3 = [et * delta / (2 * s), delta / (2 * sqrt(s)), 0,
          pz * delta / (2 * s)]
    k4 = [k1[i] + k2[i] - k3[i] for i in range(4)]
    return [k1, k2, k3, k4], delta / (2 * s)


def exact(x):
    return sp.simplify(x)


def check_zero(values, label):
    residuals = [exact(x) for x in values]
    if any(x != 0 for x in residuals):
        raise AssertionError(("FORMULA_REFERENCE_FAILED", label, residuals))
    return [str(x) for x in residuals]


def main():
    report = {"status": "REFERENCE_VALIDATION_RUNNING",
              "runtime": {"python": platform.python_version(), "sympy": sp.__version__,
                          "mpmath": mp.__version__, "mpmath_dps": 100}, "paths": []}
    mp.mp.dps = 100
    for sigma in [-1, 1]:
        vectors, phase = path(T, sigma, sp.sqrt)
        inv = [exact(dot(vectors[a], vectors[b])) for a, b in order]
        dinv = [exact(sp.diff(x, T)) for x in inv]  # Differentiate BEFORE T=2.
        values, derivs = [[exact(x.subs(T, 2)) for x in seq] for seq in [inv, dinv]]
        if sigma == -1:
            ref, dref = [9, 3, 6, 6, 12, 9], [Q(18, 5), Q(8, 5), 2, 2, Q(8, 5), Q(18, 5)]
            wref, dwref = 1 / (240 * sp.pi**4), 197 / (36000 * sp.pi**4)
            mref, dmref = 81*L**2 + 36*R**2 - 27*L*R, Q(324, 5)*L**2 + 24*R**2 - Q(72, 5)*L*R
        else:
            ref = [1, Q(1, 11), Q(10, 11), Q(10, 11), Q(100, 11), 1]
            dref = [-Q(2, 5), -Q(8, 121), -Q(202, 605), -Q(202, 605), -Q(8, 121), -Q(2, 5)]
            wref, dwref = 1 / (880 * sp.pi**4), 469 / (484000 * sp.pi**4)
            mref = L**2 + Q(100, 121)*R**2 - Q(9, 11)*L*R
            dmref = -Q(4, 5)*L**2 - Q(808, 1331)*R**2 + Q(72, 121)*L*R
        matrix = L**2 * inv[0]*inv[5] + R**2 * inv[2]*inv[3] - L*R*9*inv[1]
        matrix_t = sp.diff(matrix, T)
        reduced_t = (2*L**2*inv[0] - 9*L*R)*dinv[0] + (2*R**2*inv[2] + 9*L*R)*dinv[2]
        omega, p2, e2 = T/2, 2*T, sp.sqrt(4*T*T + 9)
        w = omega*p2**2*phase / (256*sp.pi**4*e2)
        terms = [sp.diff(omega,T)*p2**2*phase/e2,
                 2*omega*p2*sp.diff(p2,T)*phase/e2,
                 omega*p2**2*sp.diff(phase,T)/e2,
                 -omega*p2**2*phase*sp.diff(e2,T)/e2**2]
        terms = [x/(256*sp.pi**4) for x in terms]
        residuals = check_zero(
            [vectors[0][i]+vectors[1][i]-vectors[2][i]-vectors[3][i] for i in range(4)]
            + [dot(k,k)-mass2 for k,mass2 in zip(vectors,[0,9,0,9])]
            + [values[i]-ref[i] for i in range(6)] + [derivs[i]-dref[i] for i in range(6)]
            + [w.subs(T,2)-wref, sp.diff(w,T).subs(T,2)-dwref,
               matrix.subs(T,2)-mref, matrix_t.subs(T,2)-dmref,
               matrix_t-reduced_t, sp.diff(w,T)-sum(terms)]
            + [inv[0]-inv[5],inv[2]-inv[3],inv[1]-inv[0]+inv[2],inv[4]-9-inv[0]+inv[2]],
            str(sigma))
        if sigma == 1:
            residuals += check_zero([mref-(L-Q(9,22)*R)**2-Q(319,484)*R**2,
                dmref+Q(4,5)*(L-Q(45,121)*R)**2+Q(7268,14641)*R**2], "signed_forms")
        mp_residuals = []
        for (a,b), ref_t in zip(order, dref):
            numeric = mp.diff(lambda t: dot(path(t,sigma,mp.sqrt)[0][a], path(t,sigma,mp.sqrt)[0][b]), mp.mpf(2))
            target = mp.mpf(str(sp.N(ref_t,105)))
            residual = abs(numeric-target)
            assert residual < mp.mpf('1e-95')
            mp_residuals.append(mp.nstr(residual,12))
        report["paths"].append({"sigma": sigma, "invariants": list(map(str,values)),
            "invariant_derivatives": list(map(str,derivs)), "W": str(exact(w.subs(T,2))),
            "W_T": str(exact(sp.diff(w,T).subs(T,2))),
            "W_T_contributions_omega_p2_phase_e2": [str(exact(x.subs(T,2))) for x in terms],
            "M_over_64GF2": str(sp.expand(matrix.subs(T,2))),
            "M_T_over_64GF2": str(sp.expand(matrix_t.subs(T,2))),
            "exact_residuals": residuals, "mpmath_derivative_absolute_residuals": mp_residuals})
    report["status"] = "CAS_REFERENCE_CONFIRMED_SYMPY_SINGLE_AXIS"
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(json.dumps({"status": "FORMULA_REFERENCE_FAILED", "error": repr(error)}))
        sys.exit(1)
