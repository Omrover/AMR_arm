"""Simplificación de trayectorias de MoveIt y tiempos por tramo (sin ROS)."""
import math
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from amr_arm_control import trajectory as tr  # noqa: E402


def lerp(a, b, n):
    return [[x + (y - x) * k / (n - 1) for x, y in zip(a, b)] for k in range(n)]


def test_straight_line_collapses_to_endpoints():
    pts = lerp([0, 0, 0, 0], [1.0, -0.5, 0.3, 0.2], 30)
    assert tr.simplify(pts, 0.01) == [0, 29]


def test_corner_is_kept():
    a, b, c = [0, 0, 0, 0], [1.0, 0, 0, 0], [1.0, 1.0, 0, 0]
    pts = lerp(a, b, 10) + lerp(b, c, 10)[1:]
    keep = tr.simplify(pts, 0.01)
    assert keep == [0, 9, 18]


def test_blend_within_tolerance_is_bounded():
    # Esquina redondeada (como el blending de TOTG): cada punto descartado queda
    # a menos de tol de la poligonal resultante.
    pts = [[math.sin(t), 1 - math.cos(t), 0, 0] for t in [i * math.pi / 2 / 40 for i in range(41)]]
    tol = 0.02
    keep = tr.simplify(pts, tol)
    assert keep[0] == 0 and keep[-1] == 40 and len(keep) < 15
    for i, j in zip(keep, keep[1:]):
        for k in range(i + 1, j):
            assert tr._dist_to_segment(pts[k], pts[i], pts[j]) <= tol + 1e-12


def test_short_inputs():
    assert tr.simplify([], 0.01) == []
    assert tr.simplify([[0, 0]], 0.01) == [0]
    assert tr.simplify([[0, 0], [1, 1]], 0.01) == [0, 1]


def test_segment_accel_matches_planned_time():
    q0, q1 = [0, 0, 0, 0], [0.5, 0.2, 0, 0]
    a = tr.segment_accel(q0, q1, 2.0, a_max=10.0)
    assert a == pytest.approx(4 * 0.5 / 4.0)
    assert tr.duration(q0, q1, a) == pytest.approx(2.0)


def test_segment_accel_is_capped():
    q0, q1 = [0, 0, 0, 0], [2.0, 0, 0, 0]
    assert tr.segment_accel(q0, q1, 0.1, a_max=1.5) == 1.5
    assert tr.segment_accel(q0, q1, 0.0, a_max=1.5) == 1.5
    assert tr.segment_accel(q0, [0, 0, 0, 0], 1.0, a_max=1.5, a_min=0.05) == 0.05


def test_synchronized_segment_is_straight_in_joint_space():
    # Con a_i proporcionales a d_i, cada junta recorre la misma fracción de su
    # tramo en todo instante: el camino es la recta que validó MoveIt.
    q0, q1 = [0.1, -0.2, 0.3, 0.0], [0.9, 0.4, -0.5, 0.25]
    a = tr.segment_accel(q0, q1, 3.0, a_max=1.5)
    acc = tr.sync_accelerations(q0, q1, a)
    T = tr.duration(q0, q1, a)
    for t in [T * k / 10 for k in range(11)]:
        frac = []
        for x0, x1, ai in zip(q0, q1, acc):
            d = abs(x1 - x0)
            if d < 1e-9:
                continue
            s = 0.5 * ai * t * t if t <= T / 2 else d - 0.5 * ai * (T - t) ** 2
            frac.append(s / d)
        assert max(frac) - min(frac) < 1e-9
