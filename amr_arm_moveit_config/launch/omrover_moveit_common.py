"""Configuración de MoveIt compartida por los launch de este paquete."""
import os

from ament_index_python.packages import get_package_share_directory
from moveit_configs_utils import MoveItConfigsBuilder


def moveit_config(optical_frames='false'):
    urdf = os.path.join(get_package_share_directory('amr_description'), 'urdf', 'amr.urdf.xacro')
    return (MoveItConfigsBuilder('omrover_mm', package_name='amr_arm_moveit_config')
            .robot_description(file_path=urdf, mappings={'optical_frames': optical_frames})
            .robot_description_semantic(file_path='config/omrover_mm.srdf')
            .robot_description_kinematics(file_path='config/kinematics.yaml')
            .joint_limits(file_path='config/joint_limits.yaml')
            .trajectory_execution(file_path='config/moveit_controllers.yaml')
            .planning_pipelines(pipelines=['ompl', 'pilz_industrial_motion_planner'],
                                default_planning_pipeline='ompl')
            .pilz_cartesian_limits(file_path='config/pilz_cartesian_limits.yaml')
            .planning_scene_monitor(publish_robot_description=False,
                                    publish_robot_description_semantic=True)
            .to_moveit_configs())
