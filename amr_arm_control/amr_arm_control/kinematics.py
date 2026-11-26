"""Cinemática del brazo de 4 GIM6010-8 con la geometría del CAD (Brazo_4GDL_URDF_v4).

Convención de juntas = tabla DH de AMR_arm (la de la calibración, arm_sign y las
poses guardadas); respecto al export del CAD: q_CAD = (q1 - pi, -q2, -q3, q4).

    DH = [q1        l1   0    pi/2;
          q2        0    l2   0;
          q3 - phi  0    -C   0;
          q4 + gam  0    0    0]        C = hypot(l3, l4), phi = acos(l4/C), gam = pi + phi

El frame 4 está en la muñeca (junta 4) con x en la dirección de los dedos y z normal
al plano del brazo. tool0 (raíz de los dedos) está en (TOOL0_ALONG, TOOL0_PERP) de ese
frame y el TCP TCP_OFFSET más adelante sobre x (gripper.xacro).

Con q1 fijo el resto es un brazo planar en el plano (u radial, v vertical) que
contiene al eje de q1. Ángulos acumulados respecto a la horizontal:

    a2 = q2,  a3 = q2 + q3 - phi,  a4 = q2 + q3 + q4 + pi   (= pitch de la herramienta)

Sin muñeca roll, la herramienta solo apunta dentro de ese plano: la IK recibe la
posición del TCP y su pitch (0 = horizontal hacia el objetivo, -pi/2 = hacia abajo).
El yaw lo pone la base holonómica. Puro Python + numpy: lo comparten arm_server,
reach_map y las pruebas.
"""
import math

import numpy as np

L1 = 0.02045 + 0.0772     # arm_base_link (base_link del CAD) -> hombro
L2 = 0.4
L3 = 0.095
L4 = 0.24525
C = math.hypot(L3, L4)
PHI = math.acos(L4 / C)
GAMMA = math.pi + PHI

# tool0 en arm_link_4 del CAD (params.xacro: tool0_x, tool0_y, tool0_yaw) expresado
# en el frame 4 (a lo largo de los dedos / perpendicular, en el plano del brazo).
_T0_X, _T0_Y, _T0_YAW = -0.17479, -0.05323, -2.772
TOOL0_ALONG = _T0_X * math.cos(_T0_YAW) + _T0_Y * math.sin(_T0_YAW)     # 0.1822
TOOL0_PERP = -_T0_X * math.sin(_T0_YAW) + _T0_Y * math.cos(_T0_YAW)     # -0.0135
TCP_OFFSET = 0.05   # tool0 -> gripper_tcp sobre x (params.xacro: gripper_tcp_x)


def wrap(a):
    return math.atan2(math.sin(a), math.cos(a))


def _dh(theta, d, a, alpha):
    ct, st, ca, sa = math.cos(theta), math.sin(theta), math.cos(alpha), math.sin(alpha)
    return np.array([[ct, -st * ca, st * sa, a * ct],
                     [st, ct * ca, -ct * sa, a * st],
                     [0.0, sa, ca, d],
                     [0.0, 0.0, 0.0, 1.0]])


def wrist(q):
    """arm_base_link -> frame 4 (muñeca, x en la dirección de los dedos)."""
    q1, q2, q3, q4 = q
    return (_dh(q1, L1, 0.0, math.pi / 2) @ _dh(q2, 0.0, L2, 0.0)
            @ _dh(q3 - PHI, 0.0, -C, 0.0) @ _dh(q4 + GAMMA, 0.0, 0.0, 0.0))


def fk(q, tcp_offset=0.0):
    """arm_base_link -> tool0 (o el TCP si tcp_offset > 0) como matriz 4x4."""
    off = np.eye(4)
    off[0, 3] = TOOL0_ALONG + tcp_offset
    off[1, 3] = TOOL0_PERP
    return wrist(q) @ off


def tool_pitch(q):
    """Pitch de la herramienta respecto a la horizontal (rad)."""
    return wrap(q[1] + q[2] + q[3] + math.pi)


def approach_pitch(q, tcp_offset=TCP_OFFSET):
    """Pitch de la herramienta respecto a la horizontal que va de la base del
    brazo hacia el TCP: es el pitch que recibe ik()."""
    t = fk(q, tcp_offset)
    x_axis, p = t[:3, 0], t[:3, 3]
    r = math.hypot(p[0], p[1])
    if r < 1e-9:
        return tool_pitch(q)
    horiz = (x_axis[0] * p[0] + x_axis[1] * p[1]) / r
    return math.atan2(x_axis[2], horiz)


def _fit(v, lo, hi, ref):
    """v o v +- 2 pi dentro de [lo, hi], el más cercano a ref (None si ninguno).
    Hace falta porque q2 y q3 tienen rangos de más de pi en un sentido."""
    best = None
    for k in (-1, 0, 1):
        c = v + 2.0 * math.pi * k
        if lo - 1e-9 <= c <= hi + 1e-9 and (best is None or abs(c - ref) < abs(best - ref)):
            best = c
    return best


def ik(x, y, z, pitch, lower, upper, seed=None, tcp_offset=TCP_OFFSET):
    """Juntas que llevan el TCP a (x, y, z) en arm_base_link con el pitch dado
    (respecto a la horizontal de la base hacia el objetivo; ver approach_pitch).

    Prueba q1 hacia el objetivo y q1 + pi (alcanzar por encima/detrás), codo arriba y
    abajo; devuelve la solución dentro de límites más cercana a seed, o None.
    """
    seed = list(seed) if seed is not None else [0.0, 0.0, 0.0, 0.0]
    along = TOOL0_ALONG + tcp_offset
    r = math.hypot(x, y)
    base = math.atan2(y, x) if r > 1e-9 else seed[0]
    best, best_cost = None, math.inf
    for q1, u, psi in ((base, r, pitch), (base + math.pi, -r, math.pi - pitch)):
        # Muñeca: el TCP menos la herramienta (a lo largo y perpendicular al pitch).
        c, s = math.cos(psi), math.sin(psi)
        wu = u - along * c + TOOL0_PERP * s
        wv = (z - L1) - along * s - TOOL0_PERP * c
        d = (wu * wu + wv * wv - L2 * L2 - C * C) / (2.0 * L2 * C)
        if abs(d) > 1.0 + 1e-12:
            continue
        d = max(-1.0, min(1.0, d))
        for delta in (math.acos(d), -math.acos(d)):
            a2 = math.atan2(wv, wu) - math.atan2(C * math.sin(delta), L2 + C * math.cos(delta))
            a3 = a2 + delta - math.pi
            raw = [wrap(q1), wrap(a2), wrap(a3 - a2 + PHI), 0.0]
            raw[3] = wrap(psi - math.pi - raw[1] - raw[2])
            q = [_fit(v, lo, hi, s0) for v, lo, hi, s0 in zip(raw, lower, upper, seed)]
            if any(v is None for v in q):
                continue
            cost = sum(abs(a - b) for a, b in zip(q, seed))
            if cost < best_cost:
                best, best_cost = q, cost
    return best
