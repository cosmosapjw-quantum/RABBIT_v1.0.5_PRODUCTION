"""Instrumented SciPy BDF for the V1 diagnostic. SCRATCH ONLY -- never tracked, never installed.

The class below subclasses the pinned SciPy BDF and overrides ``_step_impl`` with a
VERBATIM copy of the pinned method into which probe lines have been inserted. Every
inserted line is marked ``#PROBE``. ``verify_verbatim()`` strips those lines and requires
the remainder to equal the pinned SciPy source exactly, so a transcription error is a
startup failure rather than a silent physics difference.

Nothing here edits the installed environment, the frozen module, or any tracked file.
"""

from __future__ import annotations

import inspect
import textwrap

import numpy as np

from scipy.integrate._ivp.bdf import (
    BDF,
    MAX_ORDER,
    NEWTON_MAXITER,
    MIN_FACTOR,
    MAX_FACTOR,
    change_D,
    solve_bdf_system,
)
from scipy.integrate._ivp.common import norm


class InstrumentedBDF(BDF):
    """BDF that records every trial, Newton outcome, Jacobian refresh and error norm."""

    def attach_probe(self, sink) -> None:
        self._sink = sink
        self._trial_index = 0
        self._step_index = 0
        # Wrap jac/lu OUTSIDE _step_impl so the copied method body stays verbatim.
        # ``self.jac is None`` stays False exactly as before, so control flow is unchanged.
        inner_jac, inner_lu = self.jac, self.lu

        def counting_jac(t, y):
            before = sink.raw_calls()
            J = inner_jac(t, y)
            sink.jacobian_event(t=t, raw_calls=sink.raw_calls() - before)
            return J

        def counting_lu(A):
            sink.lu_event()
            return inner_lu(A)

        self.jac, self.lu = counting_jac, counting_lu

    # ---- verbatim copy of the pinned SciPy BDF._step_impl, plus #PROBE lines ----
    def _step_impl(self):
        t = self.t
        D = self.D

        max_step = self.max_step
        min_step = 10 * np.abs(np.nextafter(t, self.direction * np.inf) - t)
        if self.h_abs > max_step:
            h_abs = max_step
            change_D(D, self.order, max_step / self.h_abs)
            self.n_equal_steps = 0
        elif self.h_abs < min_step:
            h_abs = min_step
            change_D(D, self.order, min_step / self.h_abs)
            self.n_equal_steps = 0
        else:
            h_abs = self.h_abs

        atol = self.atol
        rtol = self.rtol
        order = self.order

        alpha = self.alpha
        gamma = self.gamma
        error_const = self.error_const

        J = self.J
        LU = self.LU
        current_jac = self.jac is None
        self._step_index += 1  #PROBE
        self._sink.step_enter(self._step_index, t, h_abs, order, min_step)  #PROBE

        step_accepted = False
        while not step_accepted:
            if h_abs < min_step:
                self._sink.trial(self._step_index, t, h_abs, order, "too_small_step")  #PROBE
                return False, self.TOO_SMALL_STEP

            h = h_abs * self.direction
            t_new = t + h

            if self.direction * (t_new - self.t_bound) > 0:
                t_new = self.t_bound
                change_D(D, order, np.abs(t_new - t) / h_abs)
                self.n_equal_steps = 0
                LU = None

            h = t_new - t
            h_abs = np.abs(h)

            y_predict = np.sum(D[:order + 1], axis=0)

            scale = atol + rtol * np.abs(y_predict)
            psi = np.dot(D[1: order + 1].T, gamma[1: order + 1]) / alpha[order]

            converged = False
            c = h / alpha[order]
            while not converged:
                if LU is None:
                    LU = self.lu(self.I - c * J)

                converged, n_iter, y_new, d = solve_bdf_system(
                    self.fun, t_new, y_predict, c, psi, LU, self.solve_lu,
                    scale, self.newton_tol)
                self._sink.newton(self._step_index, t_new, h_abs, order, converged, n_iter, current_jac)  #PROBE

                if not converged:
                    if current_jac:
                        break
                    J = self.jac(t_new, y_predict)
                    LU = None
                    current_jac = True

            if not converged:
                factor = 0.5
                h_abs *= factor
                change_D(D, order, factor)
                self.n_equal_steps = 0
                LU = None
                self._sink.trial(self._step_index, t_new, h_abs / factor, order, "newton_failure")  #PROBE
                continue

            safety = 0.9 * (2 * NEWTON_MAXITER + 1) / (2 * NEWTON_MAXITER
                                                       + n_iter)

            scale = atol + rtol * np.abs(y_new)
            error = error_const[order] * d
            error_norm = norm(error / scale)
            self._sink.error_test(self._step_index, t_new, h_abs, order, error, scale, error_norm, n_iter)  #PROBE

            if error_norm > 1:
                factor = max(MIN_FACTOR,
                             safety * error_norm ** (-1 / (order + 1)))
                h_abs *= factor
                change_D(D, order, factor)
                self.n_equal_steps = 0
                # As we didn't have problems with convergence, we don't
                # reset LU here.
                self._sink.trial(self._step_index, t_new, h_abs / factor, order, "error_test_failure")  #PROBE
            else:
                step_accepted = True
                self._sink.trial(self._step_index, t_new, h_abs, order, "accepted")  #PROBE
                self._sink.accepted(self._step_index, t_new, h_abs, order, y_new, error_norm, n_iter)  #PROBE

        self.n_equal_steps += 1

        self.t = t_new
        self.y = y_new

        self.h_abs = h_abs
        self.J = J
        self.LU = LU

        # Update differences. The principal relation here is
        # D^{j + 1} y_n = D^{j} y_n - D^{j} y_{n - 1}. Keep in mind that D
        # contained difference for previous interpolating polynomial and
        # d = D^{k + 1} y_n. Thus this elegant code follows.
        D[order + 2] = d - D[order + 1]
        D[order + 1] = d
        for i in reversed(range(order + 1)):
            D[i] += D[i + 1]

        if self.n_equal_steps < order + 1:
            self._sink.step_exit(self._step_index, self.h_abs, self.order, "hold")  #PROBE
            return True, None

        if order > 1:
            error_m = error_const[order - 1] * D[order]
            error_m_norm = norm(error_m / scale)
        else:
            error_m_norm = np.inf

        if order < MAX_ORDER:
            error_p = error_const[order + 1] * D[order + 2]
            error_p_norm = norm(error_p / scale)
        else:
            error_p_norm = np.inf

        error_norms = np.array([error_m_norm, error_norm, error_p_norm])
        with np.errstate(divide='ignore'):
            factors = error_norms ** (-1 / np.arange(order, order + 3))

        delta_order = np.argmax(factors) - 1
        order += delta_order
        self.order = order

        factor = min(MAX_FACTOR, safety * np.max(factors))
        self.h_abs *= factor
        change_D(D, order, factor)
        self.n_equal_steps = 0
        self.LU = None
        self._sink.step_exit(self._step_index, self.h_abs, self.order, "reorder", error_norms.tolist(), float(factor))  #PROBE

        return True, None


def verify_verbatim() -> dict:
    """Strip #PROBE lines from our _step_impl and require equality with pinned SciPy."""

    ours = textwrap.dedent(inspect.getsource(InstrumentedBDF._step_impl))
    theirs = textwrap.dedent(inspect.getsource(BDF._step_impl))
    stripped = "\n".join(
        line for line in ours.split("\n") if "#PROBE" not in line
    )
    ok = stripped.strip() == theirs.strip()
    return {
        "verbatim_ok": ok,
        "probe_lines_removed": ours.count("#PROBE"),
        "pinned_lines": len(theirs.strip().split("\n")),
        "stripped_lines": len(stripped.strip().split("\n")),
    }
