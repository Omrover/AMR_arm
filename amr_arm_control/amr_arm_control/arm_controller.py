"""Controladores para MoveIt 2 sobre el bridge del MCU (sin ros2_control).

    arm_controller/follow_joint_trajectory   control_msgs/FollowJointTrajectory
    gripper_controller/gripper_cmd           control_msgs/GripperCommand

El firmware mueve cada GIM6010-8 con metas de posición y perfil trapezoidal
(0xDA), no con trayectorias muestreadas. La trayectoria de MoveIt se reduce a
sus vértices en el espacio articular (trajectory.simplify) y cada tramo se
ejecuta como un punto a punto sincronizado: con aceleraciones proporcionales
al recorrido de cada junta, todas avanzan en la misma fracción del tramo y el
camino es la recta que MoveIt validó contra colisiones. El brazo se detiene
en cada vértice; el tiempo de cada tramo sigue al plan mientras no exceda la
aceleración máxima.

Habla con amr_mqtt_bridge (robot real, emulador) o gz_mcu (Gazebo) por
llc/arm_command y llc/set_gripper. No usar a la vez que arm_server.
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
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectoryPoint
from control_msgs.action import FollowJointTrajectory, GripperCommand

from amr_interfaces.msg import ArmStatus
from amr_interfaces.srv import ArmCommand, SetGripper

from . import trajectory as traj

STATUS_QOS = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL,
                        reliability=ReliabilityPolicy.RELIABLE)
FJT = FollowJointTrajectory.Result


class Abort(Exception):
    def __init__(self, code, message):
        super().__init__(message)
        self.code = code


def _t(duration):
    return duration.sec + duration.nanosec * 1e-9


class ArmController(Node):
    def __init__(self):
        super().__init__('arm_controller')
        d = self.declare_parameter
        d('arm_joints', ['arm_joint_1', 'arm_joint_2', 'arm_joint_3', 'arm_joint_4'])
        d('arm_lower', [0.0, -0.06, -3.4907, -1.2217])   # CAD (convención DH)
        d('arm_upper', [2 * math.pi, 3.4907, 0.1, 1.5708])      # q3 <= 0.1: pose de calibración
        d('max_accel', 1.5)            # rad/s^2: tope por tramo (igual a joint_limits.yaml)
        d('min_accel', 0.05)
        d('path_tolerance', 0.02)      # rad: desviación máx. al simplificar la trayectoria
        d('start_tolerance', 0.05)     # rad: el primer punto debe coincidir con el brazo
        d('joint_tolerance', 0.02)     # rad para dar por alcanzado cada vértice
        d('goal_tolerance', 0.02)      # rad al final
        d('timeout_margin', 3.0)       # s sobre la duración estimada de cada tramo
        d('gripper_joint', 'gripper_finger_joint')
        d('gripper_travel', 0.05)     # m por dedo: posición del joint con el gripper abierto
        d('gripper_max_current', 1.0)  # A si GripperCommand.max_effort <= 0
        d('gripper_timeout', 3.0)
        g = lambda n: self.get_parameter(n).value  # noqa: E731
        self.joints = list(g('arm_joints'))
        self.lower, self.upper = list(g('arm_lower')), list(g('arm_upper'))

        self.lock = threading.Lock()
        self.busy = threading.Lock()
        self.q = None
        self.status = None
        cb = ReentrantCallbackGroup()
        self.create_subscription(JointState, 'joint_states', self.on_joints,
                                 qos_profile_sensor_data, callback_group=cb)
        self.create_subscription(ArmStatus, 'arm/status', self.on_status, STATUS_QOS,
                                 callback_group=cb)
        self.arm_cli = self.create_client(ArmCommand, 'llc/arm_command', callback_group=cb)
        self.grip_cli = self.create_client(SetGripper, 'llc/set_gripper', callback_group=cb)
        self.fjt = ActionServer(
            self, FollowJointTrajectory, 'arm_controller/follow_joint_trajectory',
            execute_callback=self.exec_trajectory, goal_callback=self.on_goal,
            cancel_callback=lambda _: CancelResponse.ACCEPT, callback_group=cb)
        self.grip = ActionServer(
            self, GripperCommand, 'gripper_controller/gripper_cmd',
            execute_callback=self.exec_gripper,
            cancel_callback=lambda _: CancelResponse.ACCEPT, callback_group=cb)
        self.get_logger().info(f'arm_controller listo: {self.joints}')

    # -------------------------------------------------------------- estado

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

    # --------------------------------------------------------- primitivas

    def call(self, client, req, timeout=3.0):
        if not client.wait_for_service(timeout_sec=timeout):
            raise Abort(FJT.INVALID_GOAL, f'sin servicio {client.srv_name}')
        fut = client.call_async(req)
        end = time.monotonic() + timeout
        while not fut.done():
            if time.monotonic() > end:
                raise Abort(FJT.INVALID_GOAL, f'{client.srv_name} no respondió')
            time.sleep(0.01)
        return fut.result()

    def ensure_enabled(self):
        with self.lock:
            enabled = self.status is not None and self.status.enabled
        if enabled:
            return
        res = self.call(self.arm_cli, ArmCommand.Request(op=ArmCommand.Request.ENABLE))
        if not res.accepted:
            raise Abort(FJT.INVALID_GOAL, f'enable rechazado: {res.message}')
        end = time.monotonic() + 2.0
        while time.monotonic() < end:
            with self.lock:
                if self.status is not None and self.status.enabled:
                    return
            time.sleep(0.02)
        raise Abort(FJT.INVALID_GOAL, 'el brazo no reporta enabled')

    def hold(self):
        """Fija la meta en la posición medida (cancelación o falla)."""
        q = self.joints_now()
        if q is None:
            return
        try:
            self.call(self.arm_cli, ArmCommand.Request(
                op=ArmCommand.Request.GOAL, positions=q,
                accelerations=[float(self.get_parameter('max_accel').value)] * len(q)))
        except Abort as e:
            self.get_logger().error(f'no se pudo detener el brazo: {e}')

    def reorder(self, trajectory):
        """Puntos de la trayectoria en el orden de self.joints."""
        names = list(trajectory.joint_names)
        missing = [j for j in self.joints if j not in names]
        if missing:
            raise Abort(FJT.INVALID_JOINTS, f'faltan juntas: {missing}')
        extra = [n for n in names if n not in self.joints]
        if extra:
            raise Abort(FJT.INVALID_JOINTS, f'juntas ajenas a este controlador: {extra}')
        idx = [names.index(j) for j in self.joints]
        pts = []
        for p in trajectory.points:
            if len(p.positions) != len(names):
                raise Abort(FJT.INVALID_GOAL, 'punto con número de posiciones incorrecto')
            pts.append((_t(p.time_from_start), [p.positions[i] for i in idx]))
        if not pts:
            raise Abort(FJT.INVALID_GOAL, 'trayectoria vacía')
        for t, q in pts:
            bad = [self.joints[i] for i, v in enumerate(q)
                   if not self.lower[i] - 1e-6 <= v <= self.upper[i] + 1e-6]
            if bad:
                raise Abort(FJT.INVALID_GOAL, f'fuera de límites en t={t:.2f}: {bad}')
        return pts

    # ------------------------------------------------------- trayectoria

    def exec_trajectory(self, goal_handle):
        result = FollowJointTrajectory.Result()
        if not self.busy.acquire(blocking=False):
            result.error_code, result.error_string = FJT.INVALID_GOAL, 'controlador ocupado'
            goal_handle.abort()
            return result
        try:
            self.follow(goal_handle)
            result.error_code = FJT.SUCCESSFUL
            goal_handle.succeed()
        except Abort as e:
            result.error_code, result.error_string = e.code, str(e)
            self.get_logger().warn(f'trayectoria: {e}')
            if goal_handle.is_cancel_requested:
                goal_handle.canceled()
            else:
                goal_handle.abort()
        finally:
            self.busy.release()
        return result

    def follow(self, goal_handle):
        p = lambda n: self.get_parameter(n).value  # noqa: E731
        pts = self.reorder(goal_handle.request.trajectory)
        q_now = self.joints_now()
        if q_now is None:
            raise Abort(FJT.INVALID_GOAL, 'sin /joint_states del brazo')
        dev = max(abs(a - b) for a, b in zip(pts[0][1], q_now))
        if pts[0][0] <= 1e-3 and dev > p('start_tolerance'):
            raise Abort(FJT.INVALID_GOAL,
                        f'el inicio de la trayectoria está a {dev:.3f} rad del brazo')
        # Si el primer punto no es t=0, el tramo inicial sale de la posición actual.
        if pts[0][0] > 1e-3:
            pts.insert(0, (0.0, q_now))
        keep = traj.simplify([q for _, q in pts], p('path_tolerance'))
        verts = [pts[i] for i in keep]
        self.get_logger().info(
            f'trayectoria: {len(pts)} puntos -> {len(verts) - 1} tramos, '
            f'{verts[-1][0]:.2f} s planeados')
        self.ensure_enabled()

        fb = FollowJointTrajectory.Feedback()
        fb.joint_names = list(self.joints)
        tol = p('joint_tolerance')
        q_prev = verts[0][1]
        t_prev = verts[0][0]
        for t_next, q_next in verts[1:]:
            if max(abs(a - b) for a, b in zip(q_prev, q_next)) < 1e-4:
                t_prev = t_next
                continue
            a = traj.segment_accel(q_prev, q_next, t_next - t_prev, p('max_accel'),
                                   p('min_accel'))
            res = self.call(self.arm_cli, ArmCommand.Request(
                op=ArmCommand.Request.GOAL, positions=list(q_next),
                accelerations=traj.sync_accelerations(q_prev, q_next, a, p('min_accel'))))
            if not res.accepted:
                raise Abort(FJT.INVALID_GOAL, f'goal rechazado por el MCU: {res.message}')
            timeout = traj.duration(q_prev, q_next, a) * 1.5 + p('timeout_margin')
            self.wait_vertex(goal_handle, fb, q_next, tol, timeout)
            q_prev, t_prev = q_next, t_next

        q = self.joints_now()
        err = max(abs(a - b) for a, b in zip(q, verts[-1][1]))
        if err > p('goal_tolerance'):
            raise Abort(FJT.GOAL_TOLERANCE_VIOLATED, f'error final {err:.3f} rad')

    def wait_vertex(self, goal_handle, fb, q_goal, tol, timeout):
        time.sleep(0.1)  # el MCU tarda un ciclo en dejar de reportar in_position
        end = time.monotonic() + timeout
        while True:
            if goal_handle.is_cancel_requested:
                self.hold()
                raise Abort(FJT.PATH_TOLERANCE_VIOLATED, 'cancelado')
            q = self.joints_now()
            with self.lock:
                in_pos = self.status is not None and all(self.status.in_position)
                enabled = self.status is not None and self.status.enabled
            if q is not None:
                fb.header.stamp = self.get_clock().now().to_msg()
                fb.desired = JointTrajectoryPoint(positions=list(q_goal))
                fb.actual = JointTrajectoryPoint(positions=list(q))
                fb.error = JointTrajectoryPoint(positions=[b - a for a, b in zip(q, q_goal)])
                goal_handle.publish_feedback(fb)
                if in_pos and all(abs(a - b) <= tol for a, b in zip(q, q_goal)):
                    return
            if not enabled:
                raise Abort(FJT.PATH_TOLERANCE_VIOLATED,
                            'el brazo dejó de estar habilitado (e-stop o falla)')
            if time.monotonic() > end:
                self.hold()
                raise Abort(FJT.PATH_TOLERANCE_VIOLATED,
                            f'no llegó al vértice en {timeout:.1f} s')
            time.sleep(0.02)

    # ----------------------------------------------------------- gripper

    def exec_gripper(self, goal_handle):
        """position: apertura del dedo en m (0 = cerrado, travel = abierto).
        max_effort: corriente máxima del ACS712 en A (<= 0: gripper_max_current)."""
        p = lambda n: self.get_parameter(n).value  # noqa: E731
        cmd = goal_handle.request.command
        travel = float(p('gripper_travel'))
        closure = 1.0 - max(0.0, min(travel, cmd.position)) / travel   # contrato: 1 = cerrado
        cur = cmd.max_effort if cmd.max_effort > 0.0 else float(p('gripper_max_current'))
        result = GripperCommand.Result()
        try:
            res = self.call(self.grip_cli, SetGripper.Request(
                position=float(closure), max_current_a=float(cur)))
            if not res.accepted:
                raise Abort(0, f'gripper rechazado: {res.message}')
        except Abort as e:
            self.get_logger().warn(str(e))
            goal_handle.abort()
            return result
        end = time.monotonic() + float(p('gripper_timeout'))
        while time.monotonic() < end and not goal_handle.is_cancel_requested:
            with self.lock:
                s = self.status
            if s is not None:
                result.position = (1.0 - s.gripper_position) * travel
                result.effort = float(s.gripper_current_a)
                result.stalled = bool(s.gripper_grasped)
                result.reached_goal = abs(s.gripper_position - closure) < 0.03
                if result.reached_goal or result.stalled:
                    goal_handle.succeed()
                    return result
                fb = GripperCommand.Feedback()
                fb.position, fb.effort = result.position, result.effort
                fb.stalled, fb.reached_goal = result.stalled, result.reached_goal
                goal_handle.publish_feedback(fb)
            time.sleep(0.05)
        if goal_handle.is_cancel_requested:
            goal_handle.canceled()
        else:
            goal_handle.abort()
        return result


def main(args=None):
    rclpy.init(args=args)
    node = ArmController()
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
