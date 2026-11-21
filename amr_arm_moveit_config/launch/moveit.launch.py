"""MoveIt 2 sobre el robot (real o simulado): move_group + arm_controller (+ RViz).

Requiere el bridge corriendo (llc_bridge.launch.py, emulator.launch.py o
amr_gazebo sim.launch.py): de ahí salen /joint_states, /robot_description y
los servicios llc/arm_command y llc/set_gripper.

    ros2 launch amr_arm_moveit_config moveit.launch.py rviz:=true
    ros2 launch amr_arm_moveit_config moveit.launch.py use_sim_time:=true optical_frames:=true
"""
import os
import sys

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, OpaqueFunction
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from omrover_moveit_common import moveit_config  # noqa: E402


def nodes(context):
    sim = LaunchConfiguration('use_sim_time').perform(context).lower() == 'true'
    cfg = moveit_config(LaunchConfiguration('optical_frames').perform(context))
    rviz_cfg = os.path.join(get_package_share_directory('amr_arm_moveit_config'), 'rviz',
                            'moveit.rviz')
    return [
        Node(package='moveit_ros_move_group', executable='move_group', output='screen',
             parameters=[cfg.to_dict(), {'use_sim_time': sim}]),
        Node(package='rviz2', executable='rviz2', name='rviz2_moveit', output='log',
             arguments=['-d', rviz_cfg],
             condition=IfCondition(LaunchConfiguration('rviz')),
             parameters=[cfg.robot_description, cfg.robot_description_semantic,
                         cfg.robot_description_kinematics, cfg.planning_pipelines,
                         cfg.joint_limits, {'use_sim_time': sim}]),
    ]


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('use_sim_time', default_value='false'),
        DeclareLaunchArgument('optical_frames', default_value='false',
                              description='igual que en el launch del bridge'),
        DeclareLaunchArgument('rviz', default_value='false'),
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(PathJoinSubstitution(
                [FindPackageShare('amr_arm_control'), 'launch', 'arm_controller.launch.py'])),
            launch_arguments={'use_sim_time': LaunchConfiguration('use_sim_time')}.items()),
        OpaqueFunction(function=nodes),
    ])
