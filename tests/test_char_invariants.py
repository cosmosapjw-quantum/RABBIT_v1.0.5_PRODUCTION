import numpy as np
from rabbit.diagnostics.char_invariants import qdot_energy, tail_fraction, sign_flip_count

def test_qdot_energy_zero():
    q = np.array([1.0, 2.0, 3.0])
    w = np.array([0.2, 0.3, 0.5])
    C = np.zeros(3)
    assert qdot_energy(q, w, C) == 0.0

def test_tail_fraction_bounds():
    q = np.array([1.0, 2.0, 3.0, 4.0])
    w = np.array([0.1, 0.2, 0.3, 0.4])
    C = np.array([1.0, -1.0, 2.0, -2.0])
    tf = tail_fraction(q, w, C, last_k=2)
    assert 0.0 <= tf <= 1.0

def test_sign_flip_count():
    x = np.array([1.0, -1.0, 2.0, -3.0])
    assert sign_flip_count(x) == 3
