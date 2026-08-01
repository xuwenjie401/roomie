from pathlib import Path

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def source_package_path(directory, filename):
    launch_path = Path(__file__).resolve()
    for parent in launch_path.parents:
        candidate = parent / "src" / "roomie" / directory / filename
        if candidate.exists():
            return str(candidate)
    for parent in launch_path.parents:
        candidate = parent / directory / filename
        if candidate.exists():
            return str(candidate)
    return str(launch_path.parents[1] / directory / filename)


def generate_launch_description():
    config_path = LaunchConfiguration("config_path")
    rviz_config = LaunchConfiguration("rviz_config")
    use_rviz = LaunchConfiguration("use_rviz")
    show_manual_rooms = LaunchConfiguration("show_manual_rooms")

    default_config = source_package_path("config", "pipeline_nvblox.yaml")
    default_rviz = source_package_path("rviz", "roomie_pipeline.rviz")

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "config_path",
                default_value=default_config,
                description="Path to the roomie pipeline parameter file.",
            ),
            DeclareLaunchArgument(
                "use_rviz",
                default_value="true",
                description="Start rviz2 with the roomie debug display.",
            ),
            DeclareLaunchArgument(
                "rviz_config",
                default_value=default_rviz,
                description="RViz config for roomie map and instance debug topics.",
            ),
            DeclareLaunchArgument(
                "show_manual_rooms",
                default_value="true",
                description="Publish saved manual room bounding boxes, when available.",
            ),
            Node(
                package="roomie",
                executable="roomie_pipeline_node",
                name="roomie_pipeline_node",
                output="screen",
                parameters=[config_path],
            ),
            Node(
                condition=IfCondition(show_manual_rooms),
                package="roomie",
                executable="roomie_offline_room_partition_ui.py",
                name="roomie_manual_room_marker_publisher",
                output="screen",
                arguments=[
                    "--pipeline-config",
                    config_path,
                    "--marker-only",
                    "--no-browser",
                ],
            ),
            Node(
                condition=IfCondition(use_rviz),
                package="rviz2",
                executable="rviz2",
                name="roomie_pipeline_rviz",
                arguments=["-d", rviz_config],
                output="screen",
            ),
        ]
    )
