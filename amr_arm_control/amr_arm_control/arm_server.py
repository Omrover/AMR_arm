"""Servidor del brazo: MoveArm, PickObject y PlaceObject sobre el bridge del MCU.

    move_arm      NAMED (poses de arm_server.yaml), JOINT o CARTESIAN (IK con pitch)
    pick_object   abrir -> pregrasp -> bajar -> cerrar (grasped) -> subir
    place_object  sobre el destino -> bajar -> abrir -> subir

Habla con amr_mqtt_bridge (o gz_mcu en simulación) por llc/arm_command y
llc/set_gripper, y sigue el movimiento con /joint_states y arm/status. Los
movimientos son punto a punto sincronizados (trajectory.py). Al cancelar, el
brazo se queda donde está.
"""
import math
import threading
import time

import rclpy
from rclpy.action import ActionServer, CancelResponse, GoalResponse
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data, QoSProfile, DurabilityPolicy, ReliabilityPolicy
from rclpy.time import Time
from sensor_msgs.msg import JointState
from tf2_ros import Buffer, TransformException, TransformListener

from amr_interfaces.action import MoveArm, PickObject, PlaceObject
from amr_interfaces.msg import ArmStatus
from amr_interfaces.srv import ArmCommand, SetGripper

from . import kinematics as kin
from . import trajectory as traj

OK, ERR_IK, ERR_LIMITS, ERR_MCU, ERR_TIMEOUT, ERR_CANCELED, ERR_GRASP = 0, 1, 2, 3, 4, 5, 6
STATUS_QOS = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL,
                        reliability=ReliabilityPolicy.RELIABLE)
TOP_DOWN = -math.pi / 2
# Poses tentativas (rad) con todos los eslabones fuera del plano del LiDAR;
# ajustarlas con la geometría real del CAD (config/arm_server.yaml).
DEFAULT_POSES = {
    'stow': [0.0, 0.865, -0.651, 1.357],        # TCP sobre el centro, 0.30 m arriba
    'dock_view': [0.0, 1.25, 0.1, 1.7],         # cámara horizontal a 0.38 m (dock)
    'scan': [math.pi, 1.469, -0.965, 0.295],    # cámara 0.3 m adelante, 0.8 rad abajo
    'tray_above': [math.pi, 1.487, 1.14, -1.055],  # TCP 0.15 m sobre la bandeja
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
        d('arm_upper', [math.pi, math.pi / 2, 2.5, 2.5])
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

    def move_joints(self, goal_handle, q_goal, scale=1.0, on_progress=None):
        if len(q_goal) != len(self.joints):
            raise Failure(ERR_LIMITS, f'se esperaban {len(self.joints)} juntas')
        bad = [self.joints[i] for i, v in enumerate(q_goal)
               if not self.lower[i] <= v <= self.upper[i]]
        if bad:
            raise Failure(ERR_LIMITS, f'fuera de límites: {bad}')
        if not self.wait_for(lambda: self.joints_now() is not None, 3.0):
            raise Failure(ERR_MCU, 'sin /joint_states del brazo')
        self.ensure_enabled()
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
            self.staged(goal_handle, PickObject.Feedback, [
                ('open', lambda: self.gripper(p('gripper_open').value)),
                ('pregrasp', lambda: self.move_joints(goal_handle, q_pre)),
                ('descend', lambda: self.move_joints(goal_handle, q_at, 0.4)),
                ('grasp', lambda: self.gripper(p('gripper_close').value, expect_grasp=True)),
                ('lift', lambda: self.move_joints(goal_handle, q_lift, 0.4)),
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
            self.staged(goal_handle, PlaceObject.Feedback, [
                ('above', lambda: self.move_joints(goal_handle, q_above)),
                ('descend', lambda: self.move_joints(goal_handle, q_at, 0.4)),
                ('release', lambda: self.gripper(p('gripper_open').value)),
                ('retreat', lambda: self.move_joints(goal_handle, q_above, 0.4)),
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
