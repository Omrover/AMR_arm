"""Servidor del brazo (MoveArm, PickObject, PlaceObject).

    ros2 launch amr_arm_control arm_server.launch.py [use_sim_time:=true require_grasp:=false]
"""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('use_sim_time', default_value='false'),
        DeclareLaunchArgument('require_grasp', default_value='true'),
        Node(package='amr_arm_control', executable='arm_server', name='arm_server',
             output='screen',
             parameters=[
                 PathJoinSubstitution([FindPackageShare('amr_mqtt_bridge'), 'config', 'arm.yaml']),
                 PathJoinSubstitution([FindPackageShare('amr_arm_control'), 'config',
                                       'arm_server.yaml']),
                 {'use_sim_time': LaunchConfiguration('use_sim_time'),
                  'require_grasp': ParameterValue(LaunchConfiguration('require_grasp'),
                                                  value_type=bool)}]),
    ])
