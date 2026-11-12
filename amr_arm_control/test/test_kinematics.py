"""FK/IK del brazo y sincronización de juntas (sin ROS)."""
import math
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from amr_arm_control import kinematics as k  # noqa: E402
from amr_arm_control import trajectory as tr  # noqa: E402

LOWER = [-math.pi, -math.pi / 2, -2.5, -2.5]   # arm.yaml / params.xacro
UPPER = [math.pi, math.pi / 2, 2.5, 2.5]


def tcp(q):
    t = k.fk(q, k.TCP_OFFSET)
    return t[:3, 3]


def test_zero_pose_matches_dh_by_hand():
    # q = 0: eslabón 2 horizontal (0.4), el 3 vuelve -C a -phi y el 4 apunta a
    # gamma - phi = pi respecto al 3.
    p = k.fk([0, 0, 0, 0])[:3, 3]
    a3 = -k.PHI
    a4 = math.pi
    expect_u = k.L2 - k.C * math.cos(a3) + k.L5 * math.cos(a4)
    expect_v = k.L1 - k.C * math.sin(a3) + k.L5 * math.sin(a4)
    assert p == pytest.approx([expect_u, 0.0, expect_v], abs=1e-12)
    assert k.tool_pitch([0, 0, 0, 0]) == pytest.approx(math.pi)


def test_tool_x_axis_is_pitch_in_vertical_plane():
    q = [0.7, 0.3, -0.4, 0.5]
    r = k.fk(q)[:3, :3]
    psi = k.tool_pitch(q)
    radial = np.array([math.cos(q[0]), math.sin(q[0]), 0.0])
    assert r[:, 0] == pytest.approx(math.cos(psi) * radial + [0, 0, math.sin(psi)], abs=1e-12)


@pytest.mark.parametrize('seed_n', range(5))
def test_ik_roundtrip_random(seed_n):
    rng = np.random.default_rng(seed_n)
    solved = 0
    for _ in range(200):
        q = [rng.uniform(lo * 0.95, hi * 0.95) for lo, hi in zip(LOWER, UPPER)]
        p = tcp(q)
        psi = k.approach_pitch(q)
        sol = k.ik(*p, psi, LOWER, UPPER, seed=q)
        assert sol is not None
        assert tcp(sol) == pytest.approx(p, abs=1e-9)
        assert math.cos(k.approach_pitch(sol) - psi) == pytest.approx(1.0, abs=1e-9)
        # Con la semilla en la solución original, la IK la recupera.
        assert np.allclose([k.wrap(a - b) for a, b in zip(sol, q)], 0.0, atol=1e-7)
        solved += 1
    assert solved == 200


def test_top_down_grasp_in_front_of_robot():
    # Pieza 0.40 m adelante del arm_base_link, 0.10 m más abajo, agarre vertical.
    q = k.ik(0.40, 0.0, -0.10, -math.pi / 2, LOWER, UPPER)
    assert q is not None
    assert tcp(q) == pytest.approx([0.40, 0.0, -0.10], abs=1e-9)
    assert k.tool_pitch(q) == pytest.approx(-math.pi / 2)
    assert q[0] == pytest.approx(0.0)


def test_unreachable_returns_none():
    assert k.ik(2.0, 0.0, 0.0, 0.0, LOWER, UPPER) is None
    # Con q2 limitado a 0 no puede bajar el codo a la pieza.
    assert k.ik(0.40, 0.0, -0.10, -math.pi / 2, LOWER, [math.pi, 0.0, 2.5, 2.5]) is None


def test_sync_accelerations_finish_together():
    q0 = [0.0, 0.2, -0.5, 1.0]
    q1 = [1.0, -0.3, 0.5, 1.0]
    acc = tr.sync_accelerations(q0, q1, 2.0)
    times = [2 * math.sqrt(abs(b - a) / ai) for a, b, ai in zip(q0, q1, acc) if abs(b - a) > 0]
    assert max(times) - min(times) < 1e-9
    assert acc[3] == pytest.approx(0.05)          # junta quieta: mínimo positivo
    assert max(acc) == pytest.approx(2.0)
    assert tr.duration(q0, q1, 2.0) == pytest.approx(2 * math.sqrt(1.0 / 2.0))
    assert tr.progress(q0, q1, [0.5, 0, 0, 1]) == pytest.approx(0.5)
    assert tr.progress(q0, q0, q0) == 1.0
