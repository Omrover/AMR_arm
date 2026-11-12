"""Recorrido de verificación del brazo (M5), en simulación o en el robot.

    ros2 run amr_arm_control arm_check [--ros-args -p use_sim_time:=true -p pick:=false]

1. Poses con nombre: stow -> scan -> dock_view -> stow.
2. Con pick:=true, pick de una pieza simulada frente al robot (pick_xyz en
   base_link) y place en la bandeja.
Compara la pose alcanzada (TF base_link -> gripper_tcp) con la pedida y termina
con una línea ARM_CHECK {json}. En el robot real: brazo libre de obstáculos y
mano en el e-stop.
"""
import json
import math
import time

import rclpy
from geometry_msgs.msg import PoseStamped
from rclpy.action import ActionClient
from rclpy.node import Node
from rclpy.time import Time
from tf2_ros import Buffer, TransformException, TransformListener

from amr_interfaces.action import MoveArm, PickObject, PlaceObject


class ArmCheck(Node):
    def __init__(self):
        super().__init__('arm_check')
        self.declare_parameter('poses', ['stow', 'scan', 'dock_view', 'stow'])
        self.declare_parameter('pick', True)
        self.declare_parameter('pick_xyz', [0.45, 0.0, 0.07])   # base_link
        self.move = ActionClient(self, MoveArm, 'move_arm')
        self.pick = ActionClient(self, PickObject, 'pick_object')
        self.place = ActionClient(self, PlaceObject, 'place_object')
        self.tf = Buffer()
        self.tf_listener = TransformListener(self.tf, self, spin_thread=True)
        self.results = []

    def run_action(self, client, goal, timeout):
        if not client.wait_for_server(timeout_sec=30.0):
            return None, 'sin servidor'
        send = client.send_goal_async(goal)
        rclpy.spin_until_future_complete(self, send, timeout_sec=10.0)
        if not send.done() or not send.result().accepted:
            return None, 'rechazado'
        res = send.result().get_result_async()
        rclpy.spin_until_future_complete(self, res, timeout_sec=timeout)
        if not res.done():
            send.result().cancel_goal_async()
            return None, 'timeout'
        return res.result().result, None

    def tcp(self):
        try:
            t = self.tf.lookup_transform('base_link', 'gripper_tcp', Time())
        except TransformException:
            return None
        p = t.transform.translation
        return [round(p.x, 4), round(p.y, 4), round(p.z, 4)]

    def record(self, step, result, err, t0, **extra):
        ok = err is None and result is not None and result.success
        entry = {'step': step, 'ok': ok, 'seconds': round(time.monotonic() - t0, 1),
                 'error': err or (None if ok else f'{result.error_code}: {result.message}'),
                 'tcp': self.tcp(), **extra}
        self.get_logger().info(json.dumps(entry))
        self.results.append(entry)
        return ok

    def main(self):
        for name in self.get_parameter('poses').value:
            t0 = time.monotonic()
            res, err = self.run_action(self.move, MoveArm.Goal(
                mode=MoveArm.Goal.NAMED, named_pose=name, velocity_scale=1.0), 60.0)
            self.record(f'move_{name}', res, err, t0)
        if not self.get_parameter('pick').value:
            return
        x, y, z = self.get_parameter('pick_xyz').value
        target = PoseStamped()
        target.header.frame_id = 'base_link'
        target.pose.position.x, target.pose.position.y, target.pose.position.z = x, y, z
        target.pose.orientation.w = 1.0
        t0 = time.monotonic()
        res, err = self.run_action(self.pick, PickObject.Goal(
            source=PickObject.Goal.FROM_POSE, object_pose=target), 120.0)
        lifted = self.tcp()
        ok = self.record('pick', res, err, t0, target=[x, y, z])
        if ok and lifted is not None:
            # Tras subir, el TCP queda sobre la pieza (xy) y lift_height arriba.
            self.results[-1]['xy_err'] = round(math.hypot(lifted[0] - x, lifted[1] - y), 4)
        t0 = time.monotonic()
        res, err = self.run_action(self.place, PlaceObject.Goal(
            target=PlaceObject.Goal.TO_TRAY), 120.0)
        self.record('place_tray', res, err, t0)
        t0 = time.monotonic()
        res, err = self.run_action(self.move, MoveArm.Goal(
            mode=MoveArm.Goal.NAMED, named_pose='stow', velocity_scale=1.0), 60.0)
        self.record('move_stow', res, err, t0)


def main(args=None):
    rclpy.init(args=args)
    node = ArmCheck()
    node.main()
    ok = all(r['ok'] for r in node.results) and bool(node.results)
    print('ARM_CHECK ' + json.dumps({'ok': ok, 'steps': node.results}), flush=True)
    node.destroy_node()
    rclpy.try_shutdown()
    raise SystemExit(0 if ok else 1)


if __name__ == '__main__':
    main()
