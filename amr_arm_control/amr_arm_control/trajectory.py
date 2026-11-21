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


def segment_accel(q_from, q_to, t_plan, a_max, a_min=0.05):
    """Aceleración de la junta más larga para recorrer el tramo en t_plan.

    Perfil triangular: d = a*T^2/4 -> a = 4*d/T^2. Se limita a [a_min, a_max]:
    si el plan pide más que a_max, el tramo tarda más de lo planeado."""
    d_max = max((abs(b - a) for a, b in zip(q_from, q_to)), default=0.0)
    if t_plan <= 0.0:
        return a_max
    return max(a_min, min(a_max, 4.0 * d_max / (t_plan * t_plan)))


def _dist_to_segment(p, a, b):
    """Distancia euclidiana (espacio articular) de p al segmento a-b."""
    ab = [y - x for x, y in zip(a, b)]
    ap = [y - x for x, y in zip(a, p)]
    den = sum(v * v for v in ab)
    t = 0.0 if den < 1e-18 else max(0.0, min(1.0, sum(u * v for u, v in zip(ap, ab)) / den))
    return math.sqrt(sum((x - (s + t * v)) ** 2 for x, s, v in zip(p, a, ab)))


def simplify(points, tol):
    """Índices de los vértices de la trayectoria (Ramer-Douglas-Peucker).

    MoveIt valida colisiones sobre la interpolación lineal en el espacio
    articular; los puntos que quedan a menos de tol (rad) de la recta entre dos
    vértices se descartan. Siempre conserva el primero y el último."""
    n = len(points)
    if n <= 2:
        return list(range(n))
    keep = {0, n - 1}
    stack = [(0, n - 1)]
    while stack:
        i, j = stack.pop()
        best, k_best = -1.0, None
        for k in range(i + 1, j):
            d = _dist_to_segment(points[k], points[i], points[j])
            if d > best:
                best, k_best = d, k
        if k_best is not None and best > tol:
            keep.add(k_best)
            stack.append((i, k_best))
            stack.append((k_best, j))
    return sorted(keep)


def progress(q_start, q_goal, q_now):
    """0..1 según la junta con más recorrido."""
    i = max(range(len(q_goal)), key=lambda k: abs(q_goal[k] - q_start[k]), default=None)
    if i is None or abs(q_goal[i] - q_start[i]) < 1e-6:
        return 1.0
    p = (q_now[i] - q_start[i]) / (q_goal[i] - q_start[i])
    return max(0.0, min(1.0, p))
