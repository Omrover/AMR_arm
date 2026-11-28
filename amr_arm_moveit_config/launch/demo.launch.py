"""Demo sin hardware: emulador del MCU + bridge + MoveIt + RViz.

    ros2 launch amr_arm_moveit_config demo.launch.py

Usa el broker local de emulator.launch.py (127.0.0.1:18883): no correrlo a la
vez que el bridge del robot real. El emulador arranca con el brazo en rest (su
tope, como el robot real al encender), libre de colisión: MoveIt planea desde ahí.
"""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, TimerAction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare


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
    ])
