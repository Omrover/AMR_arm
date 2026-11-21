"""Servidor del brazo (MoveArm, PickObject, PlaceObject).

    ros2 launch amr_arm_control arm_server.launch.py [use_sim_time:=true require_grasp:=false]

Con use_moveit:=true (por defecto) planea con MoveIt y además lanza move_group
y arm_controller (amr_arm_moveit_config/moveit.launch.py). Con use_moveit:=false
manda punto a punto directo al bridge, sin revisar colisiones.
"""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    use_moveit = LaunchConfiguration('use_moveit')
    return LaunchDescription([
        DeclareLaunchArgument('use_sim_time', default_value='false'),
        DeclareLaunchArgument('require_grasp', default_value='true'),
        DeclareLaunchArgument('use_moveit', default_value='true'),
        DeclareLaunchArgument('optical_frames', default_value='false',
                              description='igual que en el launch del bridge'),
        DeclareLaunchArgument('rviz', default_value='false'),
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(PathJoinSubstitution(
                [FindPackageShare('amr_arm_moveit_config'), 'launch', 'moveit.launch.py'])),
            condition=IfCondition(use_moveit),
            launch_arguments={'use_sim_time': LaunchConfiguration('use_sim_time'),
                              'optical_frames': LaunchConfiguration('optical_frames'),
                              'rviz': LaunchConfiguration('rviz')}.items()),
        Node(package='amr_arm_control', executable='arm_server', name='arm_server',
             output='screen',
             parameters=[
                 PathJoinSubstitution([FindPackageShare('amr_mqtt_bridge'), 'config', 'arm.yaml']),
                 PathJoinSubstitution([FindPackageShare('amr_arm_control'), 'config',
                                       'arm_server.yaml']),
                 {'use_sim_time': LaunchConfiguration('use_sim_time'),
                  'use_moveit': ParameterValue(use_moveit, value_type=bool),
                  'require_grasp': ParameterValue(LaunchConfiguration('require_grasp'),
                                                  value_type=bool)}]),
    ])
