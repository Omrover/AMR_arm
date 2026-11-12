"""Cinemática del brazo de 4 GIM6010-8 (tabla DH de AMR_arm/README.md).

    DH = [q1        l1   0    pi/2;
          q2        0    l2   0;
          q3 - phi  0    -C   0;
          q4 + gam  0    l5   0]        C = hypot(l3, l4), phi = acos(l4/C), gam = pi + phi

Con q1 fijo el resto es un brazo planar en el plano (u radial, v vertical) que
contiene al eje de q1. Ángulos acumulados respecto a la horizontal:

    a2 = q2,  a3 = q2 + q3 - phi,  a4 = q2 + q3 + q4 + pi   (= pitch de la herramienta)

Sin muñeca roll, la herramienta solo apunta dentro de ese plano: la IK recibe la
posición del TCP y su pitch (-pi/2 = vertical hacia abajo). El yaw lo pone la base
holonómica. Puro Python + numpy: lo comparten arm_server y las pruebas.
"""
import math

import numpy as np

L1 = 0.031 + 0.03885
L2 = 0.4
L3 = 0.095
L4 = 0.245
L5 = 0.050
C = math.hypot(L3, L4)
PHI = math.acos(L4 / C)
GAMMA = math.pi + PHI
TCP_OFFSET = 0.065   # tool0 -> gripper_tcp sobre x (gripper.xacro)


def wrap(a):
    return math.atan2(math.sin(a), math.cos(a))


def _dh(theta, d, a, alpha):
    ct, st, ca, sa = math.cos(theta), math.sin(theta), math.cos(alpha), math.sin(alpha)
    return np.array([[ct, -st * ca, st * sa, a * ct],
                     [st, ct * ca, -ct * sa, a * st],
                     [0.0, sa, ca, d],
                     [0.0, 0.0, 0.0, 1.0]])


def fk(q, tcp_offset=0.0):
    """arm_base_link -> tool0 (o el TCP si tcp_offset > 0) como matriz 4x4."""
    q1, q2, q3, q4 = q
    t = (_dh(q1, L1, 0.0, math.pi / 2) @ _dh(q2, 0.0, L2, 0.0)
         @ _dh(q3 - PHI, 0.0, -C, 0.0) @ _dh(q4 + GAMMA, 0.0, L5, 0.0))
    if tcp_offset:
        off = np.eye(4)
        off[0, 3] = tcp_offset
        t = t @ off
    return t


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


def ik(x, y, z, pitch, lower, upper, seed=None, tcp_offset=TCP_OFFSET):
    """Juntas que llevan el TCP a (x, y, z) en arm_base_link con el pitch dado
    (respecto a la horizontal de la base hacia el objetivo; ver approach_pitch).

    Prueba q1 hacia el objetivo y q1 + pi (alcanzar por detrás), codo arriba y
    abajo; devuelve la solución dentro de límites más cercana a seed, o None.
    """
    seed = list(seed) if seed is not None else [0.0, 0.0, 0.0, 0.0]
    reach = L5 + tcp_offset
    r = math.hypot(x, y)
    base = math.atan2(y, x) if r > 1e-9 else seed[0]
    best, best_cost = None, math.inf
    for q1, u, psi in ((base, r, pitch), (base + math.pi, -r, math.pi - pitch)):
        # Muñeca: el TCP menos el último eslabón en la dirección del pitch.
        wu = u - reach * math.cos(psi)
        wv = (z - L1) - reach * math.sin(psi)
        d = (wu * wu + wv * wv - L2 * L2 - C * C) / (2.0 * L2 * C)
        if abs(d) > 1.0 + 1e-12:
            continue
        d = max(-1.0, min(1.0, d))
        for delta in (math.acos(d), -math.acos(d)):
            a2 = math.atan2(wv, wu) - math.atan2(C * math.sin(delta), L2 + C * math.cos(delta))
            a3 = a2 + delta - math.pi
            q = [wrap(q1), wrap(a2), wrap(a3 - a2 + PHI), 0.0]
            q[3] = wrap(psi - math.pi - q[1] - q[2])
            if any(not lo - 1e-9 <= v <= hi + 1e-9 for v, lo, hi in zip(q, lower, upper)):
                continue
            cost = sum(abs(wrap(a - b)) for a, b in zip(q, seed))
            if cost < best_cost:
                best, best_cost = q, cost
    return best
