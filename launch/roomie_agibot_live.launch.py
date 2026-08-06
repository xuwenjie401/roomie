"""Run the current Roomie mapping pipeline on live_connect RGB-D topics."""

from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration


def generate_launch_description():
    roomie_share = Path(get_package_share_directory("roomie"))
    mapping_launch = roomie_share / "launch" / "roomie_agibot_head_mapping.launch.py"
    pipeline_config = LaunchConfiguration("pipeline_config")
    color_topic = LaunchConfiguration("color_topic")
    depth_topic = LaunchConfiguration("depth_topic")

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "pipeline_config",
                default_value=str(roomie_share / "config" / "pipeline_genie_live.yaml"),
                description="Roomie pipeline configuration for the genie_live data lineage.",
            ),
            DeclareLaunchArgument(
                "color_topic",
                default_value="/live_connect/head_color",
                description="live_connect RGB image topic consumed by the adapter.",
            ),
            DeclareLaunchArgument(
                "depth_topic",
                default_value="/live_connect/head_depth",
                description="live_connect depth image topic consumed by the adapter.",
            ),
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(str(mapping_launch)),
                launch_arguments={
                    "pipeline_config": pipeline_config,
                    "color_topic": color_topic,
                    "depth_topic": depth_topic,
                }.items(),
            )
        ]
    )
