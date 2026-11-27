"""FK/IK del brazo y sincronización de juntas (sin ROS)."""
import math
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from amr_arm_control import kinematics as k  # noqa: E402
from amr_arm_control import trajectory as tr  # noqa: E402

LOWER = [0.0, -0.06, -3.4907, -1.2217]   # arm.yaml / params.xacro (CAD)
UPPER = [2 * math.pi, 3.4907, 0.1, 1.5708]
CAD_URDF = os.path.join(os.path.dirname(__file__), '..', '..', 'models', 'urdf',
                        'Brazo_4GDL_URDF_v4.urdf')


def tcp(q):
    t = k.fk(q, k.TCP_OFFSET)
    return t[:3, 3]


def test_zero_pose_matches_dh_by_hand():
    # q = 0: eslabón 2 horizontal (0.4), el 3 vuelve -C a -phi y los dedos apuntan a
    # pi (hacia atrás); tool0 está TOOL0_ALONG sobre ellos y TOOL0_PERP a un lado.
    p = k.fk([0, 0, 0, 0])[:3, 3]
    a3 = -k.PHI
    a4 = math.pi
    expect_u = (k.L2 - k.C * math.cos(a3) + k.TOOL0_ALONG * math.cos(a4)
                - k.TOOL0_PERP * math.sin(a4))
    expect_v = (k.L1 - k.C * math.sin(a3) + k.TOOL0_ALONG * math.sin(a4)
                + k.TOOL0_PERP * math.cos(a4))
    assert p == pytest.approx([expect_u, 0.0, expect_v], abs=1e-12)
    assert k.TOOL0_ALONG == pytest.approx(0.1822, abs=1e-4)
    assert k.TOOL0_PERP == pytest.approx(-0.0135, abs=1e-4)
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


def test_horizontal_grasp_in_front_of_robot():
    # Pieza sobre una mesa de 0.60 m, 0.30 m adentro del borde (reach_map): 0.50 m
    # adelante del arm_base_link y 0.36 m arriba; gripper horizontal. El codo queda
    # por encima (q1 = pi y q2 pasa de la vertical).
    q = k.ik(0.50, 0.0, 0.3645, 0.0, LOWER, UPPER)
    assert q is not None
    assert tcp(q) == pytest.approx([0.50, 0.0, 0.3645], abs=1e-9)
    assert q[1] > math.pi / 2
    assert k.approach_pitch(q) == pytest.approx(0.0, abs=1e-9)
    assert all(lo - 1e-9 <= v <= hi + 1e-9 for v, lo, hi in zip(q, LOWER, UPPER))


def test_over_branch_has_tray_above_shape():
    # Pieza al frente sobre una mesa: la rama 'over' (la de arm_server) pone la junta 1
    # al lado contrario (q1 = pi) y el brazo por encima, como tray_above.
    q = k.ik(0.50, 0.0, 0.3645, 0.0, LOWER, UPPER, seed=[math.pi, 1.149, -0.211, -0.938],
             branch='over')
    assert q is not None and q[0] == pytest.approx(math.pi)
    assert q[1] > math.pi / 2 and q[2] < 0 and q[3] < 0
    assert k.ik(0.50, 0.0, 0.3645, 0.0, LOWER, UPPER, branch='direct') is None


def test_limits_beyond_pi():
    # q2 = 3.0 (por encima de la vertical) es válido en el CAD; la IK lo recupera
    # aunque atan2 lo entregue como 3.0 - 2 pi.
    q = [0.0, 3.0, -2.0, 0.5]
    sol = k.ik(*tcp(q), k.approach_pitch(q), LOWER, UPPER, seed=q)
    assert sol == pytest.approx(q, abs=1e-7)


@pytest.mark.skipif(not os.path.exists(CAD_URDF), reason='sin el export del CAD')
def test_matches_cad_export():
    """La cinemática reproduce el URDF exportado de SolidWorks con
    q_CAD = (q1 - pi, -q2, -q3, q4) y tool0 sobre arm_link_4."""
    import xml.etree.ElementTree as ET

    def rpy(r, p_, y):
        cr, sr, cp, sp, cy, sy = (math.cos(r), math.sin(r), math.cos(p_), math.sin(p_),
                                  math.cos(y), math.sin(y))
        return (np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
                @ np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
                @ np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]]))

    def h(rot, xyz):
        t = np.eye(4)
        t[:3, :3], t[:3, 3] = rot, xyz
        return t

    joints = {j.get('name'): j for j in ET.parse(CAD_URDF).getroot().findall('joint')}

    def cad_fk(qc):
        t = np.eye(4)
        for name, v in zip(('joint_1', 'joint_2', 'joint_3', 'joint_4'), qc):
            o, ax = joints[name].find('origin'), joints[name].find('axis')
            xyz = [float(a) for a in o.get('xyz').split()]
            t = t @ h(rpy(*[float(a) for a in o.get('rpy').split()]), xyz)
            axis = np.array([float(a) for a in ax.get('xyz').split()])
            kx = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]],
                           [-axis[1], axis[0], 0]])
            t = t @ h(np.eye(3) + math.sin(v) * kx + (1 - math.cos(v)) * kx @ kx, [0, 0, 0])
        return t @ h(rpy(0, 0, -2.772), [-0.17479, -0.05323, 0.0])

    rng = np.random.default_rng(3)
    for _ in range(200):
        q = rng.uniform(-math.pi, math.pi, 4)
        t_cad = cad_fk([q[0] - math.pi, -q[1], -q[2], q[3]])
        assert np.allclose(t_cad, k.fk(q), atol=1e-4)


def test_unreachable_returns_none():
    assert k.ik(2.0, 0.0, 0.0, 0.0, LOWER, UPPER) is None
    # Con q2 limitado a 90° el codo no puede pasar por encima de la pieza.
    assert k.ik(0.50, 0.0, 0.3645, 0.0, LOWER, [math.pi, math.pi / 2, 0.1, 1.5708]) is None


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


STOW = [0.0, 0.865, -0.651, 1.357]
DOCK_VIEW = [math.pi, 0.056, 0.087, -0.186]


def test_stow_is_higher_than_dock_view():
    assert k.min_height(STOW) > k.min_height(DOCK_VIEW) + 0.05
    # La altura no depende de q1.
    assert k.min_height([2.0] + STOW[1:]) == pytest.approx(k.min_height(STOW))


def test_rotate_first_from_high_to_low():
    # stow -> dock_view: gira con la forma de stow y luego baja.
    assert k.rotate_first_mid(STOW, DOCK_VIEW, 0.5) == [DOCK_VIEW[0]] + STOW[1:]


def test_raise_first_from_low_to_high():
    # dock_view -> stow: sube con q1 de salida y luego gira.
    assert k.rotate_first_mid(DOCK_VIEW, STOW, 0.5) == [DOCK_VIEW[0]] + STOW[1:]


def test_no_mid_for_small_turns_or_pure_rotation():
    assert k.rotate_first_mid(STOW, [0.3] + STOW[1:3] + [1.0], 0.5) is None
    assert k.rotate_first_mid(STOW, [math.pi] + STOW[1:], 0.5) is None
    assert k.rotate_first_mid(STOW, DOCK_VIEW, 0.0) is None
