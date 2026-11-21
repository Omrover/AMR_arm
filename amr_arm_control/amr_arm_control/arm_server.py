"""Servidor del brazo: MoveArm, PickObject y PlaceObject sobre el bridge del MCU.

    move_arm      NAMED (poses de arm_server.yaml), JOINT o CARTESIAN (IK con pitch)
    pick_object   abrir -> pregrasp -> bajar -> cerrar (grasped) -> subir
    place_object  sobre el destino -> bajar -> abrir -> subir

Con use_moveit (por defecto) cada movimiento se planea y ejecuta con
move_group: los traslados con OMPL (libres de colisión) y las bajadas/subidas
de pick y place en línea recta con Pilz LIN. La IK analítica (posición del TCP
+ pitch) sigue dando las metas en juntas, exactas para 4 GDL. La ejecución la
hace arm_controller sobre el bridge. Con use_moveit:=false manda punto a punto
sincronizado directo por llc/arm_command (trajectory.py), sin revisar colisiones.

El gripper va directo por llc/set_gripper; el estado se sigue con
/joint_states y arm/status. Al cancelar, el brazo se queda donde está.
"""
import math
import threading
import time

import rclpy
from rclpy.action import ActionClient, ActionServer, CancelResponse, GoalResponse
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data, QoSProfile, DurabilityPolicy, ReliabilityPolicy
from rclpy.time import Time
from sensor_msgs.msg import JointState
from tf2_ros import Buffer, TransformException, TransformListener

from moveit_msgs.action import MoveGroup
from moveit_msgs.msg import (Constraints, JointConstraint, MotionPlanRequest, MoveItErrorCodes,
                             RobotState)
from moveit_msgs.srv import GetStateValidity

from amr_interfaces.action import MoveArm, PickObject, PlaceObject
from amr_interfaces.msg import ArmStatus
from amr_interfaces.srv import ArmCommand, SetGripper

from . import kinematics as kin
from . import trajectory as traj

OK, ERR_IK, ERR_LIMITS, ERR_MCU, ERR_TIMEOUT, ERR_CANCELED, ERR_GRASP = 0, 1, 2, 3, 4, 5, 6
# MoveItErrorCodes -> error_code de las acciones. Sin plan (colisión, IK de
# Pilz, tiempo) cuenta como ERR_IK: la meta existe pero no es alcanzable.
# Las constantes de los mensajes ROS viven en la metaclase.
MOVEIT_NAMES = {getattr(MoveItErrorCodes, k): k for k in dir(type(MoveItErrorCodes))
                if k.isupper() and isinstance(getattr(MoveItErrorCodes, k, None), int)}
MOVEIT_ERRORS = {getattr(MoveItErrorCodes, k): v for k, v in (
    ('CONTROL_FAILED', ERR_MCU), ('INVALID_MOTION_PLAN', ERR_MCU),
    ('MOTION_PLAN_INVALIDATED_BY_ENVIRONMENT_CHANGE', ERR_MCU), ('PREEMPTED', ERR_CANCELED),
    ('START_STATE_IN_COLLISION', ERR_LIMITS), ('START_STATE_INVALID', ERR_LIMITS),
) if hasattr(MoveItErrorCodes, k)}
STATUS_QOS = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL,
                        reliability=ReliabilityPolicy.RELIABLE)
TOP_DOWN = -math.pi / 2
# Poses tentativas (rad) con todos los eslabones fuera del plano del LiDAR;
# ajustarlas con la geometría real del CAD (config/arm_server.yaml).
DEFAULT_POSES = {
    'stow': [0.0, 0.865, -0.651, 1.357],        # TCP sobre el centro, 0.30 m arriba
    'dock_view': [0.0, 1.46, 0.02, 1.57],       # cámara casi horizontal a 0.48 m (dock)
    'scan': [math.pi, 1.469, -0.965, 0.295],    # cámara 0.3 m adelante, 0.8 rad abajo
    'tray_above': [math.pi, 0.06, -0.4, 1.912],    # TCP 0.15 m sobre la bandeja
}


def rotate(q, v):
    """Rota v con el cuaternión q = (x, y, z, w)."""
    x, y, z, w = q
    # v' = v + 2w (u x v) + 2 u x (u x v), u = (x, y, z)
    ux = (y * v[2] - z * v[1], z * v[0] - x * v[2], x * v[1] - y * v[0])
    uux = (y * ux[2] - z * ux[1], z * ux[0] - x * ux[2], x * ux[1] - y * ux[0])
    return [v[i] + 2.0 * w * ux[i] + 2.0 * uux[i] for i in range(3)]


class Failure(Exception):
    def __init__(self, code, message):
        super().__init__(message)
        self.code = code


class ArmServer(Node):
    def __init__(self):
        super().__init__('arm_server')
        d = self.declare_parameter
        d('arm_joints', ['arm_joint_1', 'arm_joint_2', 'arm_joint_3', 'arm_joint_4'])
        d('arm_lower', [-math.pi, -math.pi / 2, -2.5, -2.5])
        d('arm_upper', [math.pi, math.pi / 2, 0.1, 2.5])   # q3 <= 0.1: el codo no cruza el eslabón 2
        d('base_frame', 'arm_base_link')
        d('tray_frame', 'tray_link')
        d('max_accel', 1.5)              # rad/s^2 con velocity_scale = 1
        d('joint_tolerance', 0.02)       # rad
        d('timeout_margin', 5.0)         # s sobre la duración estimada
        d('pose_names', list(DEFAULT_POSES))
        d('pregrasp_height', 0.08)       # m sobre la pieza
        d('lift_height', 0.12)
        d('place_clearance', 0.01)       # m sobre la superficie al soltar
        d('gripper_open', 0.0)
        d('gripper_close', 1.0)
        d('gripper_max_current', 1.0)    # A (ACS712)
        d('gripper_timeout', 3.0)
        d('require_grasp', True)         # false en simulación (gz_mcu no detecta pieza)
        d('use_moveit', True)            # false: punto a punto directo, sin colisiones
        d('planning_group', 'arm')
        d('planning_time', 5.0)          # s por intento
        d('planning_attempts', 3)
        d('goal_joint_tolerance', 0.005)  # rad en la meta de MoveIt
        d('linear_planner', 'LIN')       # Pilz: bajadas y subidas rectas
        d('gripper_joint', 'gripper_finger_joint')   # arm.yaml
        d('gripper_travel', 0.035)       # m: el joint con el gripper abierto (arm.yaml)
        g = lambda n: self.get_parameter(n).value  # noqa: E731
        self.joints = list(g('arm_joints'))
        self.lower, self.upper = list(g('arm_lower')), list(g('arm_upper'))
        self.poses = {}
        for n in g('pose_names'):
            name = f'poses.{n}'
            if not self.has_parameter(name):
                d(name, DEFAULT_POSES.get(n, [0.0] * len(self.joints)))
            self.poses[n] = [float(v) for v in g(name)]

        self.lock = threading.Lock()
        self.busy = threading.Lock()     # una acción a la vez
        self.q = None
        self.status = None
        cb = ReentrantCallbackGroup()
        self.create_subscription(JointState, 'joint_states', self.on_joints,
                                 qos_profile_sensor_data, callback_group=cb)
        self.create_subscription(ArmStatus, 'arm/status', self.on_status, STATUS_QOS,
                                 callback_group=cb)
        self.arm_cli = self.create_client(ArmCommand, 'llc/arm_command', callback_group=cb)
        self.grip_cli = self.create_client(SetGripper, 'llc/set_gripper', callback_group=cb)
        self.move_group = ActionClient(self, MoveGroup, 'move_action', callback_group=cb)
        self.validity_cli = self.create_client(GetStateValidity, 'check_state_validity',
                                               callback_group=cb)
        self.tf = Buffer()
        self.tf_listener = TransformListener(self.tf, self, spin_thread=True)
        for name, typ, fn in (('move_arm', MoveArm, self.exec_move),
                              ('pick_object', PickObject, self.exec_pick),
                              ('place_object', PlaceObject, self.exec_place)):
            ActionServer(self, typ, name, execute_callback=fn, goal_callback=self.on_goal,
                         cancel_callback=lambda _: CancelResponse.ACCEPT, callback_group=cb)
        self.get_logger().info(f'brazo listo: poses {sorted(self.poses)}')

    # ------------------------------------------------------------- estado

    def on_joints(self, msg):
        idx = {n: i for i, n in enumerate(msg.name)}
        if all(j in idx for j in self.joints):
            with self.lock:
                self.q = [msg.position[idx[j]] for j in self.joints]

    def on_status(self, msg):
        with self.lock:
            self.status = msg

    def joints_now(self):
        with self.lock:
            return list(self.q) if self.q is not None else None

    def on_goal(self, _):
        return GoalResponse.REJECT if self.busy.locked() else GoalResponse.ACCEPT

    # -------------------------------------------------------- primitivas

    def call(self, client, req, timeout=3.0):
        if not client.wait_for_service(timeout_sec=timeout):
            raise Failure(ERR_MCU, f'sin servicio {client.srv_name}')
        fut = client.call_async(req)
        end = time.monotonic() + timeout
        while not fut.done():
            if time.monotonic() > end:
                raise Failure(ERR_MCU, f'{client.srv_name} no respondió')
            time.sleep(0.01)
        return fut.result()

    def wait_for(self, predicate, timeout, goal_handle=None):
        end = time.monotonic() + timeout
        while time.monotonic() < end:
            if goal_handle is not None and goal_handle.is_cancel_requested:
                raise Failure(ERR_CANCELED, 'cancelado')
            if predicate():
                return True
            time.sleep(0.02)
        return False

    def ensure_enabled(self):
        with self.lock:
            enabled = self.status is not None and self.status.enabled
        if enabled:
            return
        res = self.call(self.arm_cli, ArmCommand.Request(op=ArmCommand.Request.ENABLE))
        if not res.accepted:
            raise Failure(ERR_MCU, f'enable rechazado: {res.message}')
        if not self.wait_for(lambda: self.status is not None and self.status.enabled, 2.0):
            raise Failure(ERR_MCU, 'el brazo no reporta enabled')

    def hold(self):
        q = self.joints_now()
        if q is not None:
            try:
                self.call(self.arm_cli, ArmCommand.Request(
                    op=ArmCommand.Request.GOAL, positions=q, accelerations=[3.0] * len(q)))
            except Failure as e:
                self.get_logger().error(f'no se pudo detener el brazo: {e}')

    def move_joints(self, goal_handle, q_goal, scale=1.0, on_progress=None, linear=False):
        """Mueve a q_goal. linear: el TCP en línea recta (Pilz LIN; sin MoveIt
        es punto a punto como cualquier otro movimiento)."""
        if len(q_goal) != len(self.joints):
            raise Failure(ERR_LIMITS, f'se esperaban {len(self.joints)} juntas')
        bad = [self.joints[i] for i, v in enumerate(q_goal)
               if not self.lower[i] <= v <= self.upper[i]]
        if bad:
            raise Failure(ERR_LIMITS, f'fuera de límites: {bad}')
        if not self.wait_for(lambda: self.joints_now() is not None, 3.0):
            raise Failure(ERR_MCU, 'sin /joint_states del brazo')
        self.ensure_enabled()
        if self.get_parameter('use_moveit').value:
            self.move_moveit(goal_handle, q_goal, scale, on_progress, linear)
            return
        q0 = self.joints_now()
        a_max = self.get_parameter('max_accel').value * max(0.05, min(1.0, scale or 1.0))
        res = self.call(self.arm_cli, ArmCommand.Request(
            op=ArmCommand.Request.GOAL, positions=list(q_goal),
            accelerations=traj.sync_accelerations(q0, q_goal, a_max)))
        if not res.accepted:
            raise Failure(ERR_MCU, f'goal rechazado: {res.message}')
        tol = self.get_parameter('joint_tolerance').value
        timeout = traj.duration(q0, q_goal, a_max) * 2.0 + self.get_parameter('timeout_margin').value

        def arrived():
            q = self.joints_now()
            if on_progress is not None and q is not None:
                on_progress(q, traj.progress(q0, q_goal, q))
            with self.lock:
                in_pos = self.status is not None and all(self.status.in_position)
            return in_pos and q is not None and all(abs(a - b) <= tol for a, b in zip(q, q_goal))

        time.sleep(0.1)  # el MCU tarda un ciclo en dejar de reportar in_position
        try:
            if not self.wait_for(arrived, timeout, goal_handle):
                raise Failure(ERR_TIMEOUT, f'no llegó en {timeout:.1f} s')
        except Failure:
            self.hold()
            raise

    def wait_future(self, fut, timeout, what):
        end = time.monotonic() + timeout
        while not fut.done():
            if time.monotonic() > end:
                raise Failure(ERR_MCU, f'{what}: sin respuesta en {timeout:.0f} s')
            time.sleep(0.01)
        return fut.result()

    def check_states(self, named):
        """Revisa con move_group que cada (nombre, q) esté libre de colisiones
        antes de mover nada: una secuencia de pick o place no arranca si alguna
        de sus poses choca. Cada pose se revisa con el gripper abierto y cerrado:
        en pick y place pasa por los dos (los dedos abiertos se abren hacia los
        lados, cerrados quedan al centro). Sin MoveIt no hace nada."""
        g = lambda n: self.get_parameter(n).value  # noqa: E731
        if not g('use_moveit'):
            return
        if not self.validity_cli.wait_for_service(timeout_sec=5.0):
            raise Failure(ERR_MCU, 'sin move_group (servicio check_state_validity)')
        names = self.joints + [g('gripper_joint')]
        # Margen: el controlador da por alcanzada una pose a joint_tolerance; una
        # pose que solo es válida en el valor exacto falla al ejecutar la siguiente.
        tol = float(g('joint_tolerance'))
        for name, q in named:
            variants = [list(q)]
            for i in range(len(q)):
                for s in (-tol, tol):
                    v = list(q)
                    v[i] += s
                    variants.append(v)
            for v in variants:
                for grip, grip_name in ((float(g('gripper_travel')), 'abierto'), (0.0, 'cerrado')):
                    req = GetStateValidity.Request(group_name=g('planning_group'))
                    req.robot_state = RobotState(joint_state=JointState(
                        name=names, position=[float(a) for a in v] + [grip]))
                    res = self.wait_future(self.validity_cli.call_async(req), 5.0,
                                           'check_state_validity')
                    if not res.valid:
                        pairs = sorted({f'{c.contact_body_1}-{c.contact_body_2}'
                                        for c in res.contacts})
                        margin = '' if v == list(q) else f' a {tol} rad de la pose'
                        raise Failure(ERR_IK, f'la pose {name} choca (gripper {grip_name}{margin}): '
                                              f'{", ".join(pairs) or "inválida"}')

    def move_moveit(self, goal_handle, q_goal, scale, on_progress, linear):
        """Planea y ejecuta con move_group hasta q_goal (meta en juntas)."""
        g = lambda n: self.get_parameter(n).value  # noqa: E731
        if not self.move_group.wait_for_server(timeout_sec=5.0):
            raise Failure(ERR_MCU, 'sin move_group (acción move_action)')
        s = max(0.05, min(1.0, scale or 1.0))
        tol = float(g('goal_joint_tolerance'))
        req = MotionPlanRequest(
            group_name=g('planning_group'), num_planning_attempts=int(g('planning_attempts')),
            allowed_planning_time=float(g('planning_time')),
            max_velocity_scaling_factor=s, max_acceleration_scaling_factor=s)
        if linear:
            req.pipeline_id = 'pilz_industrial_motion_planner'
            req.planner_id = g('linear_planner')
        req.goal_constraints.append(Constraints(joint_constraints=[
            JointConstraint(joint_name=j, position=float(v), tolerance_above=tol,
                            tolerance_below=tol, weight=1.0)
            for j, v in zip(self.joints, q_goal)]))
        goal = MoveGroup.Goal(request=req)
        goal.planning_options.plan_only = False
        q0 = self.joints_now()
        gh = self.wait_future(self.move_group.send_goal_async(goal), 10.0, 'move_group')
        if not gh.accepted:
            raise Failure(ERR_MCU, 'move_group rechazó la meta')
        res = gh.get_result_async()
        # Holgado: move_group vigila la ejecución; esto solo cubre que se caiga.
        a_max = g('max_accel') * s
        limit = (float(g('planning_time')) * int(g('planning_attempts')) +
                 3.0 * traj.duration(q0, q_goal, a_max) + 30.0)
        end = time.monotonic() + limit
        while not res.done():
            if goal_handle is not None and goal_handle.is_cancel_requested:
                gh.cancel_goal_async()
                self.hold()
                raise Failure(ERR_CANCELED, 'cancelado')
            if time.monotonic() > end:
                gh.cancel_goal_async()
                self.hold()
                raise Failure(ERR_TIMEOUT, f'move_group no terminó en {limit:.0f} s')
            q = self.joints_now()
            if on_progress is not None and q is not None:
                on_progress(q, traj.progress(q0, q_goal, q))
            time.sleep(0.05)
        code = res.result().result.error_code.val
        if code != MoveItErrorCodes.SUCCESS:
            what = 'LIN' if linear else 'OMPL'
            raise Failure(MOVEIT_ERRORS.get(code, ERR_IK),
                          f'MoveIt ({what}): {MOVEIT_NAMES.get(code, code)}')

    def ik_for(self, pose, pitch, seed=None):
        """PoseStamped -> juntas (posición del TCP + pitch)."""
        frame = self.get_parameter('base_frame').value
        p = pose.pose.position
        point = [p.x, p.y, p.z]
        if pose.header.frame_id and pose.header.frame_id != frame:
            try:
                tf = self.tf.lookup_transform(frame, pose.header.frame_id, Time())
            except TransformException as e:
                raise Failure(ERR_IK, f'sin TF {frame}<-{pose.header.frame_id}: {e}')
            r, t = tf.transform.rotation, tf.transform.translation
            point = rotate((r.x, r.y, r.z, r.w), point)
            point = [point[0] + t.x, point[1] + t.y, point[2] + t.z]
        q = kin.ik(*point, pitch, self.lower, self.upper, seed=seed or self.joints_now())
        if q is None:
            raise Failure(ERR_IK, f'sin solución IK para {[round(v, 3) for v in point]} '
                                  f'con pitch {pitch:.2f}')
        return q

    def offset(self, pose, dz):
        """Copia de la pose desplazada dz en z de su propio frame."""
        out = type(pose)()
        out.header = pose.header
        out.pose.position.x = pose.pose.position.x
        out.pose.position.y = pose.pose.position.y
        out.pose.position.z = pose.pose.position.z + dz
        out.pose.orientation = pose.pose.orientation
        return out

    def gripper(self, position, expect_grasp=False):
        g = self.get_parameter
        res = self.call(self.grip_cli, SetGripper.Request(
            position=float(position), max_current_a=float(g('gripper_max_current').value)))
        if not res.accepted:
            raise Failure(ERR_MCU, f'gripper rechazado: {res.message}')

        def done():
            with self.lock:
                s = self.status
            if s is None:
                return False
            if expect_grasp and s.gripper_grasped:
                return True
            return abs(s.gripper_position - position) < 0.03

        if not self.wait_for(done, g('gripper_timeout').value):
            raise Failure(ERR_TIMEOUT, 'el gripper no llegó')
        if expect_grasp and g('require_grasp').value:
            with self.lock:
                grasped = self.status.gripper_grasped
            if not grasped:
                raise Failure(ERR_GRASP, 'el gripper cerró sin detectar pieza (ACS712)')

    def tray_pose(self):
        from geometry_msgs.msg import PoseStamped
        p = PoseStamped()
        p.header.frame_id = self.get_parameter('tray_frame').value
        p.pose.orientation.w = 1.0
        return p

    # ----------------------------------------------------------- acciones

    def run(self, goal_handle, result, body):
        if not self.busy.acquire(blocking=False):
            result.success, result.error_code, result.message = False, ERR_MCU, 'brazo ocupado'
            goal_handle.abort()
            return result
        try:
            body()
            result.success, result.error_code, result.message = True, OK, 'ok'
            goal_handle.succeed()
        except Failure as e:
            result.success, result.error_code, result.message = False, e.code, str(e)
            self.get_logger().warn(f'{type(result).__name__}: {e}')
            if e.code == ERR_CANCELED:
                goal_handle.canceled()
            else:
                goal_handle.abort()
        finally:
            self.busy.release()
        return result

    def exec_move(self, goal_handle):
        g = goal_handle.request
        fb = MoveArm.Feedback()

        def progress(q, p):
            fb.current_joints, fb.progress = list(q), float(p)
            goal_handle.publish_feedback(fb)

        def body():
            if g.mode == MoveArm.Goal.NAMED:
                if g.named_pose not in self.poses:
                    raise Failure(ERR_LIMITS, f'pose desconocida {g.named_pose!r}')
                q = self.poses[g.named_pose]
            elif g.mode == MoveArm.Goal.JOINT:
                q = list(g.joint_positions)
            elif g.mode == MoveArm.Goal.CARTESIAN:
                q = self.ik_for(g.target, g.approach_pitch)
            else:
                raise Failure(ERR_LIMITS, f'modo desconocido {g.mode}')
            self.move_joints(goal_handle, q, g.velocity_scale, progress)

        return self.run(goal_handle, MoveArm.Result(), body)

    def staged(self, goal_handle, fb_type, stages):
        fb = fb_type()
        for i, (name, fn) in enumerate(stages):
            fb.stage, fb.progress = name, i / len(stages)
            goal_handle.publish_feedback(fb)
            fn()

    def exec_pick(self, goal_handle):
        g = goal_handle.request
        p = self.get_parameter

        def body():
            target = self.tray_pose() if g.source == PickObject.Goal.FROM_TRAY else g.object_pose
            pre = self.offset(target, p('pregrasp_height').value)
            lift = self.offset(target, p('lift_height').value)
            # Las tres poses se resuelven antes de mover: si alguna no tiene IK,
            # el brazo ni siquiera arranca.
            q_pre = self.ik_for(pre, TOP_DOWN)
            q_at = self.ik_for(target, TOP_DOWN, seed=q_pre)
            q_lift = self.ik_for(lift, TOP_DOWN, seed=q_at)
            self.check_states([('pregrasp', q_pre), ('grasp', q_at), ('lift', q_lift)])
            self.staged(goal_handle, PickObject.Feedback, [
                ('open', lambda: self.gripper(p('gripper_open').value)),
                ('pregrasp', lambda: self.move_joints(goal_handle, q_pre)),
                ('descend', lambda: self.move_joints(goal_handle, q_at, 0.4, linear=True)),
                ('grasp', lambda: self.gripper(p('gripper_close').value, expect_grasp=True)),
                ('lift', lambda: self.move_joints(goal_handle, q_lift, 0.4, linear=True)),
            ])

        return self.run(goal_handle, PickObject.Result(), body)

    def exec_place(self, goal_handle):
        g = goal_handle.request
        p = self.get_parameter

        def body():
            target = self.tray_pose() if g.target == PlaceObject.Goal.TO_TRAY else g.place_pose
            above = self.offset(target, p('pregrasp_height').value)
            at = self.offset(target, p('place_clearance').value)
            q_above = self.ik_for(above, TOP_DOWN)
            q_at = self.ik_for(at, TOP_DOWN, seed=q_above)
            self.check_states([('above', q_above), ('place', q_at)])
            self.staged(goal_handle, PlaceObject.Feedback, [
                ('above', lambda: self.move_joints(goal_handle, q_above)),
                ('descend', lambda: self.move_joints(goal_handle, q_at, 0.4, linear=True)),
                ('release', lambda: self.gripper(p('gripper_open').value)),
                ('retreat', lambda: self.move_joints(goal_handle, q_above, 0.4, linear=True)),
            ])

        return self.run(goal_handle, PlaceObject.Result(), body)


def main(args=None):
    rclpy.init(args=args)
    node = ArmServer()
    ex = MultiThreadedExecutor(num_threads=4)
    ex.add_node(node)
    try:
        ex.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
