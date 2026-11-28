"""Mapa de alcance del brazo con el pitch de la herramienta fijo.

Recorre el plano vertical frente al robot (q1 = 0, eje x de la base) y, para cada
punto (distancia, altura desde el piso), resuelve la IK del arm_server con el pitch
dado y revisa que los eslabones no entren al chasis ni al LiDAR. Un punto es
"tomable" si además tienen IK sin choque el pregrasp y el lift de la secuencia de
pick con ese pitch:

    pitch 0      (gripper paralelo al piso): pregrasp = approach m ANTES en x
    pitch -pi/2  (vertical, hacia abajo):    pregrasp = approach m ARRIBA

    python3 -m amr_arm_control.reach_map --pitch 0 --png alcance.png
    python3 -m amr_arm_control.reach_map --pitch -90 --mount-x 0.14 --mount-z 0.246

La geometría por defecto es la de params.xacro (TBD hasta el URDF del CAD);
pasar --mount-* / --front / --top con las medidas reales.
"""
import argparse
import math

import numpy as np

from amr_arm_control import kinematics as kin

LOWER = [0.0, -0.06, -3.4907, -1.2217]      # arm.yaml / params.xacro (CAD)
UPPER = [2 * math.pi, 3.4907, 0.1, 1.5708]
# Semillas para que la IK pruebe codo arriba y codo abajo, al frente y por encima.
SEEDS = ([0.0, 0.6, -1.6, 0.0], [0.0, 0.1, -0.6, 0.0], [0.0, 1.2, -2.2, 0.0],
         [math.pi, 2.5, -0.5, 0.0], [math.pi, 1.6, -0.3, -1.0])

UNREACH, COLLIDE, REACH, PICK = 0, 1, 2, 3


def chain(q, tcp=kin.TCP_OFFSET):
    """Base, hombro, codo, muñeca, tool0 y TCP, en arm_base_link."""
    q1, q2, q3, q4 = q
    t = np.eye(4)
    pts = [t[:3, 3].copy()]
    for dh in ((q1, kin.L1, 0.0, math.pi / 2), (q2, 0.0, kin.L2, 0.0),
               (q3 - kin.PHI, 0.0, -kin.C, 0.0)):
        t = t @ kin._dh(*dh)
        pts.append(t[:3, 3].copy())
    pts.append(kin.fk(q)[:3, 3].copy())
    pts.append(kin.fk(q, tcp)[:3, 3].copy())
    return pts


class Robot:
    """Cajas de choque en el plano x-z del robot (z desde el piso)."""

    def __init__(self, a):
        self.mx, self.mz = a.mount_x, a.mount_z
        self.branch = getattr(a, 'branch', None) or None
        self.lower = getattr(a, 'lower', None) or LOWER
        self.upper = getattr(a, 'upper', None) or UPPER
        r = a.clearance
        # (x_min, x_max, z_max): chasis y LiDAR, inflados por el radio de los eslabones
        self.boxes = [(-a.back - r, a.front + r, a.top + r),
                      (a.lidar_x - 0.04 - r, a.lidar_x + 0.04 + r, a.lidar_top + r)]
        self.table = None

    def set_table(self, edge, height):
        """Mesa desde x = edge hacia adelante con la superficie a height (None: sin mesa).
        Se revisa contra los eslabones, no contra el gripper (que toma la pieza sobre
        ella); por eso no se infla: el codo y la muñeca deben quedar por encima."""
        self.table = None if edge is None else (edge, math.inf, height)

    def collides(self, q):
        pts = [(self.mx + p[0], self.mz + p[2]) for p in chain(q)]
        # desde el codo (pts[2]) hasta el TCP: el hombro está montado sobre el chasis.
        # El último segmento (pts[4] -> TCP) es el gripper.
        segs = list(zip(pts[2:], pts[3:]))
        for k, ((x0, z0), (x1, z1)) in enumerate(segs):
            boxes = self.boxes
            if self.table is not None and k < len(segs) - 1:
                boxes = boxes + [self.table]
            for s in np.linspace(0.0, 1.0, 12):
                x, z = x0 + s * (x1 - x0), z0 + s * (z1 - z0)
                if z < 0.01:
                    return True                     # piso
                for bx0, bx1, bz in boxes:
                    if bx0 <= x <= bx1 and z <= bz:
                        return True
        return False

    def solve(self, x_floor, z_floor, pitch):
        """-> (estado, q) para el TCP en (x, z) del robot."""
        u, z = x_floor - self.mx, z_floor - self.mz
        found = None
        for seed in SEEDS:
            q = kin.ik(u, 0.0, z, pitch, self.lower, self.upper, seed=seed, branch=self.branch)
            if q is None:
                continue
            if not self.collides(q):
                return REACH, q
            found = q
        return (COLLIDE, found) if found is not None else (UNREACH, None)


def pick_ok(robot, x, z, pitch, a):
    if robot.solve(x, z, pitch)[0] != REACH:
        return False
    if abs(pitch) < math.radians(45):
        # pregrasp: approach m antes sobre la radial desde la base del brazo
        back = a.approach if x >= robot.mx else -a.approach
        pre = robot.solve(x - back, z, pitch)[0]
    else:
        pre = robot.solve(x, z + a.approach, pitch)[0]
    return pre == REACH and robot.solve(x, z + a.lift, pitch)[0] == REACH


def tables(robot, pitch, a, label):
    """Para cada altura de mesa: profundidad tomable medida desde el borde de la mesa.

    El robot se estaciona con el frente a table_gap del borde; la pieza se toma con
    el TCP a obj_half sobre la superficie y los eslabones no pueden bajar de ella.
    """
    edge = a.front + a.table_gap
    print(f'\n=== {label}: mesa con el borde a {a.table_gap:.2f} m del frente del robot, '
          f'TCP a {a.obj_half:.2f} m sobre la superficie ===')
    print('altura mesa | pieza tomable (distancia desde el borde de la mesa)')
    rows = []
    for h in np.arange(a.table_min, a.table_max + 1e-9, 0.05):
        robot.set_table(edge, h)
        ok = [x for x in np.arange(edge, edge + 1.0, a.step)
              if pick_ok(robot, x, h + a.obj_half, pitch, a)]
        robot.set_table(None, 0.0)
        if ok:
            gaps = len(ok) < round((ok[-1] - ok[0]) / a.step) + 1
            print(f'  {h:4.2f} m    | {ok[0] - edge:4.2f} … {ok[-1] - edge:4.2f} m'
                  f'{"  (con huecos)" if gaps else ""}')
        else:
            print(f'  {h:4.2f} m    | —')
        rows.append((h, ok))
    return rows


def build(robot, pitch, a):
    xs = np.arange(a.x_min, a.x_max + 1e-9, a.step)
    zs = np.arange(0.0, a.z_max + 1e-9, a.step)
    grid = np.zeros((len(zs), len(xs)), dtype=int)
    for i, z in enumerate(zs):
        for j, x in enumerate(xs):
            st, _ = robot.solve(x, z, pitch)
            if st == REACH and pick_ok(robot, x, z, pitch, a):
                st = PICK
            grid[i, j] = st
    return xs, zs, grid


def summarize(xs, zs, grid, a, label):
    print(f'\n=== {label}: TCP en el plano frontal (q1 = 0) ===')
    print('altura TCP  | x tomable (desde el centro del robot) | más allá del frente (+{:.2f})'
          .format(a.front))
    for i, z in enumerate(zs):
        if round(z / a.step) % max(1, round(0.05 / a.step)):
            continue
        cols = np.where(grid[i] == PICK)[0]
        if cols.size == 0:
            print(f'  {z:5.2f} m   |  —')
            continue
        x0, x1 = xs[cols[0]], xs[cols[-1]]
        beyond = x1 - a.front
        print(f'  {z:5.2f} m   |  {x0:+.2f} … {x1:+.2f} m'
              f'{"" if cols.size == cols[-1] - cols[0] + 1 else " (con huecos)":14s}|'
              f'  {"hasta %.2f m" % beyond if beyond > 0 else "no sale del frente"}')
    front = grid[:, xs > a.front]
    rows = np.where((front == PICK).any(axis=1))[0]
    if rows.size:
        print(f'  -> frente al robot se puede tomar entre {zs[rows[0]]:.2f} y '
              f'{zs[rows[-1]]:.2f} m de altura')
    else:
        print('  -> no hay puntos tomables frente al robot')


def plot(results, a, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    from matplotlib.patches import Patch, Rectangle

    cmap = ListedColormap(['#f2f2f2', '#f4b6b6', '#b9d7f0', '#2b7bba'])
    fig, axes = plt.subplots(1, len(results), figsize=(6.2 * len(results), 5.6), squeeze=False)
    for ax, (label, xs, zs, grid) in zip(axes[0], results):
        h = a.step / 2
        ax.imshow(grid, origin='lower', cmap=cmap, vmin=0, vmax=3, interpolation='nearest',
                  extent=(xs[0] - h, xs[-1] + h, zs[0] - h, zs[-1] + h))
        ax.add_patch(Rectangle((-a.back, 0.0), a.back + a.front, a.top, fill=False, lw=1.5,
                               ec='#333'))
        ax.add_patch(Rectangle((a.lidar_x - 0.04, a.top), 0.08, a.lidar_top - a.top,
                               fill=False, lw=1.2, ec='#333'))
        ax.plot([a.mount_x], [a.mount_z], 'k^', ms=8)
        ax.axvline(a.front, color='#333', lw=0.8, ls='--')
        ax.set_title(label)
        ax.set_xlabel('x desde el centro del robot [m]')
        ax.set_ylabel('altura del TCP desde el piso [m]')
        ax.set_aspect('equal')
        ax.grid(alpha=0.25)
    fig.legend(handles=[Patch(color='#2b7bba', label='tomable (pregrasp + agarre + lift)'),
                        Patch(color='#b9d7f0', label='alcanzable, sin secuencia completa'),
                        Patch(color='#f4b6b6', label='IK sí, pero choca con la base'),
                        Patch(color='#f2f2f2', label='fuera de alcance')],
               loc='lower center', ncol=4, fontsize=9)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(path, dpi=110)
    print(f'\nimagen: {path}')


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--pitch', type=float, nargs='+', default=[0.0, -90.0],
                    help='grados respecto a la horizontal (0 = paralelo al piso, -90 = hacia abajo)')
    ap.add_argument('--mount-x', type=float, default=0.1194, help='arm_base_link en x [m] (medido)')
    ap.add_argument('--mount-z', type=float, default=0.196,
                    help='arm_base_link (cara inferior de la placa, sobre el techo) desde el piso [m]')
    ap.add_argument('--front', type=float, default=0.26, help='frente del chasis [m] (CAD)')
    ap.add_argument('--back', type=float, default=0.33, help='parte trasera del chasis [m]')
    ap.add_argument('--top', type=float, default=0.20,
                    help='cubierta del chasis desde el piso [m] (CAD)')
    ap.add_argument('--lower', type=float, nargs=4, default=None,
                    help='límites inferiores q1..q4 [rad] (por defecto los del CAD)')
    ap.add_argument('--upper', type=float, nargs=4, default=None)
    ap.add_argument('--branch', choices=['over', 'direct'], default=None,
                    help="rama de la IK: 'over' = forma de tray_above (como arm_server)")
    ap.add_argument('--lidar-x', type=float, default=0.3045,
                    help='centro de la torre del LiDAR (CAD: x 0.264..0.345)')
    ap.add_argument('--lidar-top', type=float, default=0.29)
    ap.add_argument('--clearance', type=float, default=0.03,
                    help='holgura de choque (radio de eslabón) [m]')
    ap.add_argument('--approach', type=float, default=0.08, help='pregrasp [m]')
    ap.add_argument('--lift', type=float, default=0.05, help='lift tras agarrar [m]')
    ap.add_argument('--x-min', type=float, default=-0.1)
    ap.add_argument('--x-max', type=float, default=1.0)
    ap.add_argument('--z-max', type=float, default=1.2)
    ap.add_argument('--step', type=float, default=0.01)
    ap.add_argument('--png', default='')
    ap.add_argument('--tables', action='store_true',
                    help='además, profundidad tomable sobre una mesa para cada altura')
    ap.add_argument('--table-gap', type=float, default=0.08,
                    help='frente del robot -> borde de la mesa [m] (el LiDAR sobresale 0.06)')
    ap.add_argument('--table-min', type=float, default=0.30)
    ap.add_argument('--table-max', type=float, default=0.95)
    ap.add_argument('--obj-half', type=float, default=0.03,
                    help='altura del TCP sobre la mesa (media altura de la pieza) [m]')
    a = ap.parse_args(argv)

    robot = Robot(a)
    results = []
    for deg in a.pitch:
        label = f'pitch {deg:+.0f}°' + (' (paralelo al piso)' if abs(deg) < 1e-6 else
                                         ' (vertical)' if abs(deg + 90) < 1e-6 else '')
        xs, zs, grid = build(robot, math.radians(deg), a)
        summarize(xs, zs, grid, a, label)
        results.append((label, xs, zs, grid))
        if a.tables:
            tables(robot, math.radians(deg), a, label)
    if a.png:
        plot(results, a, a.png)


if __name__ == '__main__':
    main()
