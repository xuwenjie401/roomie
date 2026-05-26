from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from launch.substitutions import PathJoinSubstitution


def generate_launch_description():
    config_path = LaunchConfiguration("config_path")

    default_config = PathJoinSubstitution(
        [
            FindPackageShare("roomie"),
            "config",
            "pipeline.yaml",
        ]
    )

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "config_path",
                default_value=default_config,
                description="Path to the roomie pipeline parameter file.",
            ),
            Node(
                package="roomie",
                executable="roomie_pipeline_node",
                name="roomie_pipeline_node",
                output="screen",
                parameters=[config_path],
            ),
        ]
    )
