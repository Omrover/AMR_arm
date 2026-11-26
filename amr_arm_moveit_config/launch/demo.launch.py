"""Demo sin hardware: emulador del MCU + bridge + MoveIt + RViz.

    ros2 launch amr_arm_moveit_config demo.launch.py

Usa el broker local de emulator.launch.py (127.0.0.1:18883): no correrlo a la
vez que el bridge del robot real. El emulador arranca con el brazo en rest (su
tope, como el robot real al encender). Ahí el eslabón 4 descansa sobre el 2 y
MoveIt no planea desde un estado en colisión: al iniciar se habilita y se abre
el codo 0.35 rad directo por llc/arm_command (lo mismo que hace arm_server).
"""
from launch import LaunchDescription
from launch.actions import (DeclareLaunchArgument, ExecuteProcess, IncludeLaunchDescription,
                            TimerAction)
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare

UNPARK = '[3.1412, -0.05, -0.292, 0.0]'   # rest con el codo abierto park_unfold
TO_UNPARK = (
    'ros2 service call /llc/arm_command amr_interfaces/srv/ArmCommand "{op: 1}" && '
    'sleep 1 && '
    'ros2 service call /llc/arm_command amr_interfaces/srv/ArmCommand '
    f'"{{op: 0, positions: {UNPARK}, accelerations: [1.0, 1.0, 1.0, 1.0]}}"')


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('rviz', default_value='true'),
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(PathJoinSubstitution(
                [FindPackageShare('amr_bringup'), 'launch', 'emulator.launch.py']))),
        TimerAction(period=3.0, actions=[IncludeLaunchDescription(
            PythonLaunchDescriptionSource(PathJoinSubstitution(
                [FindPackageShare('amr_arm_moveit_config'), 'launch', 'moveit.launch.py'])),
            launch_arguments={'optical_frames': 'true',
                              'rviz': LaunchConfiguration('rviz')}.items())]),
        TimerAction(period=5.0, actions=[
            ExecuteProcess(cmd=['bash', '-c', TO_UNPARK], name='arm_unpark', output='log')]),
    ])
