"""Controladores FollowJointTrajectory y GripperCommand para MoveIt 2.

    ros2 launch amr_arm_control arm_controller.launch.py [use_sim_time:=true]

Usa los mismos nombres de juntas y límites que el bridge (arm.yaml).
"""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('use_sim_time', default_value='false'),
        Node(package='amr_arm_control', executable='arm_controller', name='arm_controller',
             output='screen',
             parameters=[
                 PathJoinSubstitution([FindPackageShare('amr_mqtt_bridge'), 'config', 'arm.yaml']),
                 PathJoinSubstitution([FindPackageShare('amr_arm_control'), 'config',
                                       'arm_controller.yaml']),
                 {'use_sim_time': LaunchConfiguration('use_sim_time')}]),
    ])
