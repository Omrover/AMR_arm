import argparse
import math

from amr_arm_control import reach_map as rm

# Geometría por defecto del mapa (CAD, montaje provisional) y límites del CAD.
GEOM = argparse.Namespace(mount_x=0.14, mount_z=0.198 + 0.0675, front=0.26, back=0.33,
                          top=0.20, lidar_x=0.3045, lidar_top=0.29, clearance=0.03,
                          approach=0.08, lift=0.05, lower=None, upper=None)
EDGE = 0.26 + 0.08          # borde de la mesa: frente del chasis + 8 cm


def test_chain_ends_at_tcp():
    q = [0.3, 0.4, -1.0, -0.5]
    tcp = rm.kin.fk(q, rm.kin.TCP_OFFSET)[:3, 3]
    assert max(abs(a - b) for a, b in zip(rm.chain(q)[-1], tcp)) < 1e-9


def test_table_pick_with_elbow_over_the_piece():
    r = rm.Robot(GEOM)
    # Mesa a 0.60 m: pieza 0.30 m adentro del borde, gripper horizontal.
    r.set_table(EDGE, 0.60)
    assert rm.pick_ok(r, EDGE + 0.30, 0.63, 0.0, GEOM)
    # Mesa a 0.45 m: cerca del borde no (q4 del CAD), al fondo sí.
    r.set_table(EDGE, 0.45)
    assert not rm.pick_ok(r, EDGE + 0.10, 0.48, 0.0, GEOM)
    assert rm.pick_ok(r, EDGE + 0.60, 0.48, 0.0, GEOM)


def test_q4_limit_is_what_blocks_low_tables():
    relaxed = argparse.Namespace(**vars(GEOM))
    relaxed.lower = [-math.pi, -0.06, -3.4907, -2.5]
    relaxed.upper = [math.pi, 3.4907, 0.1, 2.5]
    cad, wide = rm.Robot(GEOM), rm.Robot(relaxed)
    for r in (cad, wide):
        r.set_table(EDGE, 0.40)
    assert not rm.pick_ok(cad, EDGE + 0.40, 0.43, 0.0, GEOM)
    assert rm.pick_ok(wide, EDGE + 0.40, 0.43, 0.0, relaxed)


def test_piece_inside_chassis_is_never_pickable():
    assert not rm.pick_ok(rm.Robot(GEOM), 0.0, 0.15, 0.0, GEOM)
