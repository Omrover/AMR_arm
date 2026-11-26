"""Demo sin hardware: emulador del MCU + bridge + MoveIt + RViz.

    ros2 launch amr_arm_moveit_config demo.launch.py

Usa el broker local de emulator.launch.py (127.0.0.1:18883): no correrlo a la
vez que el bridge del robot real. El emulador arranca con el brazo en q = 0; al
iniciar se habilita y se lleva a stow directo por llc/arm_command.
"""
from launch import LaunchDescription
from launch.actions import (DeclareLaunchArgument, ExecuteProcess, IncludeLaunchDescription,
                            TimerAction)
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare

STOW = '[0.0, 0.865, -0.651, 1.357]'   # = group_state stow del SRDF
TO_STOW = (
    'ros2 service call /llc/arm_command amr_interfaces/srv/ArmCommand "{op: 1}" && '
    'sleep 1 && '
    'ros2 service call /llc/arm_command amr_interfaces/srv/ArmCommand '
    f'"{{op: 0, positions: {STOW}, accelerations: [1.0, 1.0, 1.0, 1.0]}}"')


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
            ExecuteProcess(cmd=['bash', '-c', TO_STOW], name='arm_to_stow', output='log')]),
    ])
