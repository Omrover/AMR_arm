"""Punto a punto sincronizado con los perfiles trapezoidales del GIM6010-8.

Cada junta recibe solo su aceleración (= desaceleración). Con perfiles
triangulares todas tardan T = 2*sqrt(d_i / a_i), así que con
a_i = a_max * d_i / d_max terminan juntas. Si el motor topa con su velocidad
máxima, la junta más larga llega un poco después: el servidor espera a que
todas reporten in_position.
"""
import math


def sync_accelerations(q_from, q_to, a_max, a_min=0.05):
    d = [abs(b - a) for a, b in zip(q_from, q_to)]
    d_max = max(d) if d else 0.0
    if d_max < 1e-6:
        return [a_max] * len(d)
    return [max(a_min, a_max * di / d_max) for di in d]


def duration(q_from, q_to, a_max):
    """Duración estimada (s) del movimiento sincronizado (perfil triangular)."""
    d_max = max((abs(b - a) for a, b in zip(q_from, q_to)), default=0.0)
    return 2.0 * math.sqrt(d_max / a_max) if a_max > 0 else math.inf


def progress(q_start, q_goal, q_now):
    """0..1 según la junta con más recorrido."""
    i = max(range(len(q_goal)), key=lambda k: abs(q_goal[k] - q_start[k]), default=None)
    if i is None or abs(q_goal[i] - q_start[i]) < 1e-6:
        return 1.0
    p = (q_now[i] - q_start[i]) / (q_goal[i] - q_start[i])
    return max(0.0, min(1.0, p))
